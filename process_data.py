import json
import requests
import pandas as pd
from culc_bpi import BMSScoreAnalyzer
from tqdm import tqdm
from dataclasses import dataclass
import pickle

with open("data/insane.json", "r", encoding="utf-8") as f:
    insane_table = json.load(f)

    # {
    #     "artist": "裏 吉 川 / obj: black train",
    #     "comment": "発狂BMS段位認定 六段4曲目",
    #     "level": "16",
    #     "lr2_bmsid": "354",
    #     "md5": "dd99972264eeab7e4cd22d9a64ccb569",
    #     "name_diff": "http://absolute.pv.land.to/",
    #     "title": "A c i - L - G O D",
    #     "url": "http://www5.pf-x.net/~malie/",
    #     "url_diff": "http://absolute.pv.land.to/"
    # },

@dataclass
class SongData:
    artist: str
    level: str
    lr2_bmsid: str
    md5: str
    title: str
    players: int
    clear_players: int
    theoretical_score: int
    top_score: int
    average_score: float
    std_score: float
    optomized_p: float
    BPI90: int
    BPI80: int
    BPI70: int
    BPI60: int
    BPI50: int
    BPI40: int
    BPI30: int
    BPI20: int
    BPI10: int

song_list = []
for sd in tqdm(insane_table):
    bmsid = sd["lr2_bmsid"]
    analyzer = BMSScoreAnalyzer(bmsid)
    analyzer.analyze()

    song_list.append(
        SongData(
            artist=sd["artist"],
            level=sd["level"],
            lr2_bmsid=sd["lr2_bmsid"],
            md5=sd["md5"],
            title=analyzer.song_title,
            players=analyzer.players,
            clear_players=analyzer.clear_players,
            theoretical_score=analyzer.theoretical_score,
            top_score=analyzer.top_score,
            average_score=analyzer.average_score,
            std_score=analyzer.std_dev_score,
            optomized_p=analyzer.optimized_p,
            BPI90=analyzer.bpi_scores[0],
            BPI80=analyzer.bpi_scores[1],
            BPI70=analyzer.bpi_scores[2],
            BPI60=analyzer.bpi_scores[3],
            BPI50=analyzer.bpi_scores[4],
            BPI40=analyzer.bpi_scores[5],
            BPI30=analyzer.bpi_scores[6],
            BPI20=analyzer.bpi_scores[7],
            BPI10=analyzer.bpi_scores[8]
        )
    )



    print(f"楽曲タイトル: {analyzer.song_title}")
    print(f"プレイヤー数: {analyzer.players}")
    print(f"クリア人数: {analyzer.clear_players}")
    print(f"理論値スコア: {analyzer.theoretical_score}")
    print(f"全国トップ: {analyzer.top_score}")
    print(f"平均スコア: {analyzer.average_score}")
    print(f"標準偏差: {analyzer.std_dev_score}")
    print(f"最適化されたp値: {analyzer.optimized_p}")
    print(f"BPI90〜10のスコア: {analyzer.bpi_scores}")

res = pd.DataFrame(song_list)

with open("data/insane.pkl", "wb") as f:
    pickle.dump(res, f)