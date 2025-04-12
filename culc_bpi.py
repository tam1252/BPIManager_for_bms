import pandas as pd
import numpy as np
import warnings
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from io import StringIO
import logging
import lxml  # lxmlパーサーを明示的にインポート
import concurrent.futures
from math import ceil, log
import math
import statistics
from scipy.optimize import minimize
# import matplotlib.pyplot as plt

# 警告を無視
warnings.filterwarnings("ignore", message="You provided Unicode markup but also provided a value for from_encoding. Your from_encoding will be ignored.")

# ロギングの設定
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

max_workers = 15  # スレッドプールの最大ワーカー数

class BMSScoreAnalyzer:
    def __init__(self, bmsid: int):
        self.bms_url = f"http://www.dream-pro.info/~lavalse/LR2IR/search.cgi?mode=ranking&bmsid={bmsid}"
        self.song_title = None
        self.players = None
        self.clear_players = None
        self.theoretical_score = None
        self.top_score = None
        self.average_score = None
        self.std_dev_score = None
        self.optimized_p = None
        self.bpi_scores = None
        self._percentiles = [0.000419851815, 0.000922273675, 0.002025926035, 0.004450280245, 0.009775773585, 0.021474096895, 0.047171390935, 0.10361973002, 0.227617804685]
        self._bpi_range = range(90, 0, -10)

    # 理論値スコアを抽出
    def _extract_theoretical_score(self, score_str: str) -> int | None:
        match = re.search(r'/(\d+)', score_str)
        if match:
            return int(match.group(1))
        return None

    # ランキングを整形 (pandas DataFrame を使用)
    def _format_ranking(self, ranking: pd.DataFrame) -> pd.DataFrame:
        valid_ranks = ["AAA", "AA", "A", "B", "C", "D", "E", "F"]
        valid_clear = ["★FULLCOMBO", "FULLCOMBO", "CLEAR", "HARD", "EASY"]
        return ranking[(ranking["ランク"].isin(valid_ranks)) & (ranking["クリア"].isin(valid_clear))]

    # 全てのプレイヤーの実際のスコアをリストで保管
    def _get_wholescore(self, ranking_data: pd.DataFrame) -> list[int]:
        return [int(match.group(1)) for score_str in ranking_data['スコア'] if (match := re.search(r'^(\d+)/', score_str))]

    # URLからBeautifulSoupオブジェクトを使用してデータを読み込む (pandas DataFrame を返す)
    def _read_url_with_bs4(self, page: int) -> pd.DataFrame:
        url = f"{self.bms_url}&page={page}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = 'shift-jis'  # 明示的にエンコーディングを設定
            soup = BeautifulSoup(response.text, 'lxml', from_encoding='shift-jis') # エンコーディングを指定
            html_string = str(soup)
            tables = pd.read_html(StringIO(html_string), encoding='shift-jis', flavor='lxml')
            if len(tables) > 3:
                ranking = tables[3]
                if 'ランク' in ranking.columns:
                    return self._format_ranking(ranking)
                else:
                    logging.error(f"ページ {page} から取得した DataFrame に 'ランク' 列がありません。")
                    logging.error(f"DataFrame の列名: {ranking.columns.tolist()}")
                    return pd.DataFrame()
            else:
                logging.error(f"ページ {page} から期待されるテーブルが見つかりませんでした。")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logging.error(f"ネットワークエラー (ページ {page}): {e}")
            return pd.DataFrame()
        except UnicodeDecodeError as e:
            logging.error(f"デコードエラー (ページ {page}): {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"予期しないエラー (ページ {page}): {e}")
            return pd.DataFrame()

    # 非同期に単一のURLからランキングデータを取得する関数 (pandas)
    def _fetch_ranking_async(self, page: int) -> pd.DataFrame | None:
        try:
            return self._read_url_with_bs4(page)
        except Exception as e:
            logging.error(f"非同期処理中にエラーが発生しました (pandas, page {page}): {e}")
            return None

    def _pgf(self, x, m):
        """PGF関数"""
        if x == 1:
            return m
        else:
            return 0.5 / (1 - x)

    def _calculate_bpi(self, s, k, z, m, p):
        """単曲BPIを計算する関数"""
        S = self._pgf(s / m, m)
        K = self._pgf(k / m, m)
        Z = self._pgf(z / m, m)
        S_prime = S / K
        Z_prime = Z / K

        if s >= k:
            return 100 * (np.log(S_prime) ** p) / (np.log(Z_prime) ** p)
        else:
            return min(-100 * (np.log(S_prime) ** p) / (np.log(Z_prime) ** p), -15)

    def _objective_function(self, p, s1, k, z, m, s2):
        """p値を最適化するための目的関数"""
        rag = 0
        for i in range(len(s1)):
            rag += (self._calculate_bpi(s2[i], k, z, m, p) - s1[i]) ** 2
        return rag

    def _optimize_p(self, s1, k, z, m, s2):
        """p値を最適化する関数"""
        result = minimize(self._objective_function, 1.0, args=(s1, k, z, m, s2), bounds=[(0.8, 1.5)])
        return result.x[0]

    def _reverse_calculate_s(self, bpi_target, k, z, m, p):
        if z == m:
            return float(round(m - (log(m - k) - (log(2 * (m - k)) * (bpi_target / 100) ** (1 / p))) ** math.e, 2))
        return float(round(m - (log(m - k) - ((log(m - z) - log(m - k)) * (bpi_target / 100) ** (1 / p))) ** math.e, 2))

    def analyze(self):
        try:
            # 最初のページの情報を取得 (同期的に players 数を取得)
            response_first = requests.get(f"{self.bms_url}&page=1")
            response_first.raise_for_status()
            response_first.encoding = 'shift-jis'

            soup = BeautifulSoup(response_first.content, "html.parser")
            song_title = soup.find("h1").text
            self.song_title = song_title

            a = pd.read_html(StringIO(response_first.text), encoding='shift-jis', flavor='lxml')
            self.players = int(a[1]["プレイ"][1])
            num_pages = ceil(self.players / 100)

            # 全てのページを非同期で取得 (pandas)
            all_rankings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._fetch_ranking_async, p): p for p in range(1, num_pages + 1)}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="処理済み"):
                    result = future.result()
                    if result is not None and not result.empty:
                        all_rankings.append(result)

            all_scores = []
            if all_rankings:
                self.theoretical_score = self._extract_theoretical_score(all_rankings[0]['スコア'].iloc[0])
                for ranking in all_rankings:
                    all_scores.extend(self._get_wholescore(ranking))

            all_scores = sorted(all_scores, reverse=True)
            self.clear_players = len(all_scores)
            if all_scores:
                self.top_score = all_scores[0]
                self.average_score = round(statistics.mean(all_scores), 4)
                self.std_dev_score = round(statistics.stdev(all_scores) , 4)

                bpi_people = [ceil(p * len(all_scores)) for p in self._percentiles]
                bpi_actual_scores = [all_scores[p] for p in bpi_people]

                # p値の最適化
                if self.average_score is not None and self.top_score is not None and self.theoretical_score is not None and bpi_actual_scores:
                    self.optimized_p = self._optimize_p(np.arange(10, 100, 10), self.average_score, self.top_score, self.theoretical_score, bpi_actual_scores)
                    self.bpi_scores = [self._reverse_calculate_s(bpi, self.average_score, self.top_score, self.theoretical_score, self.optimized_p) for bpi in self._bpi_range]
                else:
                    logging.warning("BPI計算に必要なデータが不足しています。")
                    self.optimized_p = None
                    self.bpi_scores = None
            else:
                logging.warning("有効なクリアデータが見つかりませんでした。")
                self.top_score = None
                self.average_score = None
                self.std_dev_score = None
                self.optimized_p = None
                self.bpi_scores = None

        except Exception as e:
            logging.error(f"分析中にエラーが発生しました: {e}")
            self.players = None
            self.clear_players = None
            self.theoretical_score = None
            self.top_score = None
            self.average_score = None
            self.std_dev_score = None
            self.optimized_p = None
            self.bpi_scores = None

if __name__ == "__main__":
    bmsid = 15
    analyzer = BMSScoreAnalyzer(bmsid)
    analyzer.analyze()

    print(f"楽曲タイトル: {analyzer.song_title}")
    print(f"プレイヤー数: {analyzer.players}")
    print(f"クリア人数: {analyzer.clear_players}")
    print(f"理論値スコア: {analyzer.theoretical_score}")
    print(f"全国トップ: {analyzer.top_score}")
    print(f"平均スコア: {analyzer.average_score}")
    print(f"標準偏差: {analyzer.std_dev_score}")
    print(f"最適化されたp値: {analyzer.optimized_p}")
    print(f"BPI90〜10のスコア: {analyzer.bpi_scores}")