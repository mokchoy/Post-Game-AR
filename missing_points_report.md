# Missing points and annotation continuity report

Audit date: **7 October 2026 (Australia/Sydney)**  
Workspace: `C:\Users\ILLEGEAR\Documents\Post-Game-AR`

## Findings

The local annotations omit **88 distinct internal rally IDs** across **35 distinct match/game combinations**. Every internal ID gap is corroborated by the neighbouring scores advancing by the corresponding number of points.

There are also **5 distinct games with a nonterminal-looking last recorded score** and **1 separate score-progression inconsistency**. These are review cases, not additional confirmed missing points.

| Dataset | Matches | Game/set CSVs | Recorded rallies | Stroke rows | Missing internal rally IDs | Games with ID gaps | Possible incomplete endings | Separate score-issue games | Games passing all continuity checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `set-1` | 44 | 104 | 3683 | 36484 | 41 | 17 | 3 | 0 | 86 |
| `set-2` | 58 | 140 | 4944 | 52356 | 50 | 21 | 3 | 1 | 116 |

The per-folder missing-ID counts total **91 annotation omissions**. Three of those refer to the same missing points in overlapping match copies, so the count of distinct missing points is **88**, not 91. The six per-folder ending-review cases similarly represent **five distinct games**. Categories overlap; a game can have both an internal gap and an uncertain ending.

## Scope and interpretation

- Read both `match.csv` files and all **244** game/set CSVs they declare. Every declared file was present.
- Group stroke rows by `rally` within each game, preserving the original rally IDs. A missing point is a missing rally group, not a missing stroke row within an existing rally.
- Check that scores advance by exactly one point between consecutive annotated rallies, allowing larger positive jumps only when corroborated by absent rally IDs.
- Interpret scores as **A–B**: A is `winner` and B is `loser` in that dataset's `match.csv`. These are match-level identities, not near/far court sides. Each case below names A and B explicitly.
- Internal omissions mean the local CSV has no rows for that rally ID. They do not establish why it was omitted, whether its video is available, or whether it can be recovered from another release.
- The outcome and exact score of an omitted point are not reconstructed. When both players' scores increase across a gap, several intervening score sequences can fit the same evidence.
- Ending checks use the conventional 21-point system applicable to these 2018–2022 matches: 21 with a two-point lead, deuce extending to 29, and a cap of 30. See [BWF Laws of Badminton, section 7](https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/Section%204.1-%20Laws%20of%20Badminton%20-%2019052018A.pdf). Retirement, an incorrect score label, or a truncated annotation can also explain a nonterminal-looking ending.
- This audit does not validate trajectories, stroke-coordinate accuracy, contact-frame alignment, or annotation completeness within a recorded rally. Flagged or incomplete stroke rows are not counted as absent points here.
- Source links open the original local CSV, with one-based physical line numbers including the header. A link for a neighbouring rally points to its first stroke row.

## Confirmed internal annotation omissions

Each missing rally ID below is absent from the local CSV and supported by the neighbouring score jump. `Pending` means the omission is documented but has not yet been recovered or assigned a cause. The same point is listed once when it occurs in both folders.

### 1. CHEN Long vs CHOU Tien Chen — World Tour Finals, Group-Stage, game 2

Dataset references: `set-1` match **2**. **A:** CHEN Long; **B:** CHOU Tien Chen.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 22 | rally 21: 9–12 | rally 23: 10–13 | Pending |

Evidence:

- Missing **22**, `set-1` match 2: [rally 21, 9–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage/set2.csv:215>) → [rally 23, 10–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage/set2.csv:226>).

### 2. CHEN Long vs CHOU Tien Chen — Denmark Open 2019, Quarter-finals, game 2

Dataset references: `set-1` match **4**. **A:** CHEN Long; **B:** CHOU Tien Chen.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 2 | rally 1: 1–0 | rally 3: 3–0 | Pending |
| 16 | rally 15: 8–7 | rally 17: 8–9 | Pending |

Evidence:

- Missing **2**, `set-1` match 4: [rally 1, 1–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal/set2.csv:2>) → [rally 3, 3–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal/set2.csv:10>).
- Missing **16**, `set-1` match 4: [rally 15, 8–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal/set2.csv:119>) → [rally 17, 8–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal/set2.csv:129>).

### 3. Kento MOMOTA vs CHOU Tien Chen — Denmark Open 2018, Finals, game 1

Dataset references: `set-1` match **6**. **A:** Kento MOMOTA; **B:** CHOU Tien Chen.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 24 | rally 23: 10–13 | rally 25: 11–14 | Pending |

Evidence:

- Missing **24**, `set-1` match 6: [rally 23, 10–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals/set1.csv:302>) → [rally 25, 11–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals/set1.csv:310>).

### 4. Kento MOMOTA vs CHOU Tien Chen — Denmark Open 2018, Finals, game 2

Dataset references: `set-1` match **6**. **A:** Kento MOMOTA; **B:** CHOU Tien Chen.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 8 | rally 7: 3–4 | rally 9: 3–6 | Pending |
| 34 | rally 33: 13–20 | rally 35: 15–20 | Pending |

Evidence:

- Missing **8**, `set-1` match 6: [rally 7, 3–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals/set2.csv:75>) → [rally 9, 3–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals/set2.csv:78>).
- Missing **34**, `set-1` match 6: [rally 33, 13–20](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals/set2.csv:311>) → [rally 35, 15–20](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals/set2.csv:315>).

### 5. Anders ANTONSEN vs Jonatan CHRISTIE — Indonesia Masters 2020, Quarter-finals, game 1

Dataset references: `set-1` match **14**. **A:** Anders ANTONSEN; **B:** Jonatan CHRISTIE.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 20 | rally 19: 11–8 | rally 21: 11–10 | Pending |

Evidence:

- Missing **20**, `set-1` match 14: [rally 19, 11–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals/set1.csv:196>) → [rally 21, 11–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals/set1.csv:201>).

### 6. Anthony Sinisuka GINTING vs Viktor AXELSEN — Indonesia Masters 2020, Semi-finals, game 1

Dataset references: `set-1` match **16**. **A:** Anthony Sinisuka GINTING; **B:** Viktor AXELSEN.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 12 | rally 11: 6–5 | rally 13: 6–7 | Pending |
| 17 | rally 16: 8–8 | rally 18: 8–10 | Pending |
| 20 | rally 19: 8–11 | rally 22: 10–12 | Pending |
| 21 | rally 19: 8–11 | rally 22: 10–12 | Pending |
| 23 | rally 22: 10–12 | rally 24: 10–14 | Pending |
| 26 | rally 25: 11–14 | rally 27: 13–14 | Pending |
| 39 | rally 38: 19–19 | rally 40: 20–20 | Pending |

Evidence:

- Missing **12**, `set-1` match 16: [rally 11, 6–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:108>) → [rally 13, 6–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:128>).
- Missing **17**, `set-1` match 16: [rally 16, 8–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:149>) → [rally 18, 8–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:157>).
- Missing **20, 21**, `set-1` match 16: [rally 19, 8–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:170>) → [rally 22, 10–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:177>).
- Missing **23**, `set-1` match 16: [rally 22, 10–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:177>) → [rally 24, 10–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:180>).
- Missing **26**, `set-1` match 16: [rally 25, 11–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:190>) → [rally 27, 13–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:193>).
- Missing **39**, `set-1` match 16: [rally 38, 19–19](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:294>) → [rally 40, 20–20](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set1.csv:303>).

### 7. Anthony Sinisuka GINTING vs Viktor AXELSEN — Indonesia Masters 2020, Semi-finals, game 2

Dataset references: `set-1` match **16**. **A:** Anthony Sinisuka GINTING; **B:** Viktor AXELSEN.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 9 | rally 8: 4–4 | rally 10: 5–5 | Pending |
| 19 | rally 18: 10–8 | rally 20: 12–8 | Pending |
| 21 | rally 20: 12–8 | rally 24: 15–9 | Pending |
| 22 | rally 20: 12–8 | rally 24: 15–9 | Pending |
| 23 | rally 20: 12–8 | rally 24: 15–9 | Pending |
| 27 | rally 26: 15–11 | rally 28: 17–11 | Pending |

Evidence:

- Missing **9**, `set-1` match 16: [rally 8, 4–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:60>) → [rally 10, 5–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:75>).
- Missing **19**, `set-1` match 16: [rally 18, 10–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:139>) → [rally 20, 12–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:143>).
- Missing **21, 22, 23**, `set-1` match 16: [rally 20, 12–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:143>) → [rally 24, 15–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:157>).
- Missing **27**, `set-1` match 16: [rally 26, 15–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:163>) → [rally 28, 17–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:171>).

### 8. NG Ka Long Angus vs Jonatan CHRISTIE — Malaysia Masters 2020, Quarter-finals, game 1

Dataset references: `set-1` match **17**. **A:** NG Ka Long Angus; **B:** Jonatan CHRISTIE.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 5 | rally 4: 2–2 | rally 6: 2–4 | Pending |
| 7 | rally 6: 2–4 | rally 8: 3–5 | Pending |
| 17 | rally 16: 6–10 | rally 18: 7–11 | Pending |
| 29 | rally 28: 11–17 | rally 30: 13–17 | Pending |

Evidence:

- Missing **5**, `set-1` match 17: [rally 4, 2–2](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:22>) → [rally 6, 2–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:26>).
- Missing **7**, `set-1` match 17: [rally 6, 2–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:26>) → [rally 8, 3–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:40>).
- Missing **17**, `set-1` match 17: [rally 16, 6–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:133>) → [rally 18, 7–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:142>).
- Missing **29**, `set-1` match 17: [rally 28, 11–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:236>) → [rally 30, 13–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set1.csv:241>).

### 9. NG Ka Long Angus vs Jonatan CHRISTIE — Malaysia Masters 2020, Quarter-finals, game 2

Dataset references: `set-1` match **17**. **A:** NG Ka Long Angus; **B:** Jonatan CHRISTIE.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 9 | rally 8: 4–4 | rally 11: 6–5 | Pending |
| 10 | rally 8: 4–4 | rally 11: 6–5 | Pending |
| 20 | rally 19: 13–6 | rally 21: 15–6 | Pending |
| 26 | rally 25: 16–9 | rally 28: 18–10 | Pending |
| 27 | rally 25: 16–9 | rally 28: 18–10 | Pending |

Evidence:

- Missing **9, 10**, `set-1` match 17: [rally 8, 4–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set2.csv:75>) → [rally 11, 6–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set2.csv:80>).
- Missing **20**, `set-1` match 17: [rally 19, 13–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set2.csv:168>) → [rally 21, 15–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set2.csv:182>).
- Missing **26, 27**, `set-1` match 17: [rally 25, 16–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set2.csv:211>) → [rally 28, 18–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set2.csv:216>).

### 10. NG Ka Long Angus vs Jonatan CHRISTIE — Malaysia Masters 2020, Quarter-finals, game 3

Dataset references: `set-1` match **17**. **A:** NG Ka Long Angus; **B:** Jonatan CHRISTIE.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 15 | rally 14: 7–7 | rally 16: 9–7 | Pending |
| 31 | rally 30: 16–14 | rally 32: 16–16 | Pending |
| 35 | rally 34: 17–17 | rally 36: 19–17 | Pending |

Evidence:

- Missing **15**, `set-1` match 17: [rally 14, 7–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set3.csv:143>) → [rally 16, 9–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set3.csv:156>).
- Missing **31**, `set-1` match 17: [rally 30, 16–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set3.csv:306>) → [rally 32, 16–16](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set3.csv:326>).
- Missing **35**, `set-1` match 17: [rally 34, 17–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set3.csv:339>) → [rally 36, 19–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals/set3.csv:344>).

### 11. Mia BLICHFELDT vs Busanan ONGBAMRUNGPHAN — YONEX THAILAND OPEN 2021, Quarter-finals, game 2

Dataset references: `set-1` match **22**. **A:** Mia BLICHFELDT; **B:** Busanan ONGBAMRUNGPHAN.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 20 | rally 19: 9–10 | rally 23: 12–11 | Pending |
| 21 | rally 19: 9–10 | rally 23: 12–11 | Pending |
| 22 | rally 19: 9–10 | rally 23: 12–11 | Pending |

Evidence:

- Missing **20, 21, 22**, `set-1` match 22: [rally 19, 9–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Mia_Blichfeldt_Busanan_Ongbamrungphan_YONEX_Thailand_Open_2021_QuarterFinals/set2.csv:148>) → [rally 23, 12–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Mia_Blichfeldt_Busanan_Ongbamrungphan_YONEX_Thailand_Open_2021_QuarterFinals/set2.csv:150>).

### 12. Anthony Sinisuka GINTING vs Rasmus GEMKE — YONEX THAILAND OPEN 2021, Quarter-finals, game 3

Dataset references: `set-1` match **24**; `set-2` match **58**. **A:** Anthony Sinisuka GINTING; **B:** Rasmus GEMKE.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 22 | rally 21: 16–5 | rally 23: 18–5 | Pending |

Evidence:

- Missing **22**, `set-1` match 24: [rally 21, 16–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals/set3.csv:200>) → [rally 23, 18–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals/set3.csv:215>).
- Missing **22**, `set-2` match 58: [rally 21, 16–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals/set3.csv:200>) → [rally 23, 18–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals/set3.csv:215>).

### 13. Viktor AXELSEN vs Anthony Sinisuka GINTING — YONEX THAILAND OPEN 2021, Semi-finals, game 3

Dataset references: `set-1` match **27**. **A:** Viktor AXELSEN; **B:** Anthony Sinisuka GINTING.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 15 | rally 14: 5–9 | rally 16: 7–9 | Pending |

Evidence:

- Missing **15**, `set-1` match 27: [rally 14, 5–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals/set3.csv:120>) → [rally 16, 7–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals/set3.csv:138>).

### 14. Viktor AXELSEN vs NG Ka Long Angus — YONEX THAILAND OPEN 2021, Finals, game 1

Dataset references: `set-1` match **28**. **A:** Viktor AXELSEN; **B:** NG Ka Long Angus.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 6 | rally 5: 4–1 | rally 7: 6–1 | Pending |

Evidence:

- Missing **6**, `set-1` match 28: [rally 5, 4–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Viktor_Axelsen_Ng_Ka_Long_Angus_YONEX_Thailand_Open_2021_Finals/set1.csv:60>) → [rally 7, 6–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Viktor_Axelsen_Ng_Ka_Long_Angus_YONEX_Thailand_Open_2021_Finals/set1.csv:80>).

### 15. Viktor AXELSEN vs LIEW Daren — TOYOTA THAILAND OPEN 2021, Quarter-finals, game 1

Dataset references: `set-1` match **33**. **A:** Viktor AXELSEN; **B:** LIEW Daren.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 23 | rally 22: 14–8 | rally 24: 16–8 | Pending |

Evidence:

- Missing **23**, `set-1` match 33: [rally 22, 14–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Viktor_Axelsen_Liew_Daren_TOYOTA_THAILAND_OPEN_2021_QuarterFinals/set1.csv:172>) → [rally 24, 16–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Viktor_Axelsen_Liew_Daren_TOYOTA_THAILAND_OPEN_2021_QuarterFinals/set1.csv:176>).

### 16. An Se Young vs Carolina MARIN — HSBC BWF WORLD TOUR FINALS 2020, Quarter-finals, game 2

Dataset references: `set-1` match **38**; `set-2` match **2**. **A:** An Se Young; **B:** Carolina MARIN.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 30 | rally 29: 11–18 | rally 31: 11–20 | Pending |

Evidence:

- Missing **30**, `set-1` match 38: [rally 29, 11–18](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set2.csv:235>) → [rally 31, 11–20](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set2.csv:243>).
- Missing **30**, `set-2` match 2: [rally 29, 11–18](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set2.csv:235>) → [rally 31, 11–20](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set2.csv:243>).

### 17. An Se Young vs Carolina MARIN — HSBC BWF WORLD TOUR FINALS 2020, Quarter-finals, game 3

Dataset references: `set-1` match **38**; `set-2` match **2**. **A:** An Se Young; **B:** Carolina MARIN.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 32 | rally 31: 18–13 | rally 33: 18–15 | Pending |

Evidence:

- Missing **32**, `set-1` match 38: [rally 31, 18–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set3.csv:193>) → [rally 33, 18–15](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set3.csv:198>).
- Missing **32**, `set-2` match 2: [rally 31, 18–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set3.csv:193>) → [rally 33, 18–15](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set3.csv:198>).

### 18. Viktor AXELSEN vs Kento MOMOTA — Malaysia Open 2022, Finals, game 1

Dataset references: `set-2` match **22**. **A:** Viktor AXELSEN; **B:** Kento MOMOTA.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 15 | rally 14: 11–3 | rally 16: 13–3 | Pending |

Evidence:

- Missing **15**, `set-2` match 22: [rally 14, 11–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kento_MOMOTA_PETRONAS_Malaysia_Open_2022_Finals/set1.csv:171>) → [rally 16, 13–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kento_MOMOTA_PETRONAS_Malaysia_Open_2022_Finals/set1.csv:180>).

### 19. Viktor AXELSEN vs Kento MOMOTA — Malaysia Open 2022, Finals, game 2

Dataset references: `set-2` match **22**. **A:** Viktor AXELSEN; **B:** Kento MOMOTA.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 5 | rally 4: 3–1 | rally 6: 5–1 | Pending |
| 24 | rally 23: 17–6 | rally 25: 19–6 | Pending |

Evidence:

- Missing **5**, `set-2` match 22: [rally 4, 3–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kento_MOMOTA_PETRONAS_Malaysia_Open_2022_Finals/set2.csv:37>) → [rally 6, 5–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kento_MOMOTA_PETRONAS_Malaysia_Open_2022_Finals/set2.csv:56>).
- Missing **24**, `set-2` match 22: [rally 23, 17–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kento_MOMOTA_PETRONAS_Malaysia_Open_2022_Finals/set2.csv:241>) → [rally 25, 19–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kento_MOMOTA_PETRONAS_Malaysia_Open_2022_Finals/set2.csv:245>).

### 20. Ratchanok INTANON vs WANG Zhi Yi — Malaysia Open 2022, Semi-finals, game 1

Dataset references: `set-2` match **24**. **A:** Ratchanok INTANON; **B:** WANG Zhi Yi.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 8 | rally 7: 4–3 | rally 9: 6–3 | Pending |
| 12 | rally 11: 7–4 | rally 14: 10–4 | Pending |
| 13 | rally 11: 7–4 | rally 14: 10–4 | Pending |
| 15 | rally 14: 10–4 | rally 16: 12–4 | Pending |
| 19 | rally 18: 14–4 | rally 20: 16–4 | Pending |
| 23 | rally 22: 17–5 | rally 24: 18–6 | Pending |
| 27 | rally 26: 19–7 | rally 28: 20–8 | Pending |

Evidence:

- Missing **8**, `set-2` match 24: [rally 7, 4–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:67>) → [rally 9, 6–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:74>).
- Missing **12, 13**, `set-2` match 24: [rally 11, 7–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:91>) → [rally 14, 10–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:95>).
- Missing **15**, `set-2` match 24: [rally 14, 10–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:95>) → [rally 16, 12–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:101>).
- Missing **19**, `set-2` match 24: [rally 18, 14–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:111>) → [rally 20, 16–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:118>).
- Missing **23**, `set-2` match 24: [rally 22, 17–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:149>) → [rally 24, 18–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:156>).
- Missing **27**, `set-2` match 24: [rally 26, 19–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:166>) → [rally 28, 20–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set1.csv:170>).

### 21. Ratchanok INTANON vs WANG Zhi Yi — Malaysia Open 2022, Semi-finals, game 2

Dataset references: `set-2` match **24**. **A:** Ratchanok INTANON; **B:** WANG Zhi Yi.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 3 | rally 2: 1–1 | rally 5: 4–1 | Pending |
| 4 | rally 2: 1–1 | rally 5: 4–1 | Pending |
| 29 | rally 28: 16–12 | rally 31: 17–14 | Pending |
| 30 | rally 28: 16–12 | rally 31: 17–14 | Pending |
| 32 | rally 31: 17–14 | rally 33: 17–16 | Pending |
| 34 | rally 33: 17–16 | rally 35: 18–17 | Pending |

Evidence:

- Missing **3, 4**, `set-2` match 24: [rally 2, 1–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:8>) → [rally 5, 4–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:19>).
- Missing **29, 30**, `set-2` match 24: [rally 28, 16–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:210>) → [rally 31, 17–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:228>).
- Missing **32**, `set-2` match 24: [rally 31, 17–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:228>) → [rally 33, 17–16](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:236>).
- Missing **34**, `set-2` match 24: [rally 33, 17–16](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:236>) → [rally 35, 18–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Ratchanok_INTANON_WANG_Zhi_Yi_Malaysia_Open_2022_SemiFinals/set2.csv:244>).

### 22. Chico Aura DWI WARDOYO vs LU Guang Zu — Malaysia Open 2022, Semi-finals, game 2

Dataset references: `set-2` match **25**. **A:** Chico Aura DWI WARDOYO; **B:** LU Guang Zu.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 6 | rally 5: 2–3 | rally 7: 2–5 | Pending |
| 25 | rally 24: 11–13 | rally 26: 13–13 | Pending |

Evidence:

- Missing **6**, `set-2` match 25: [rally 5, 2–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Chico_Aura_DWI_WARDOYO_LU_Guang_Zu_PERODUA_Malaysia_Masters_2022_Semifinals/set2.csv:27>) → [rally 7, 2–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Chico_Aura_DWI_WARDOYO_LU_Guang_Zu_PERODUA_Malaysia_Masters_2022_Semifinals/set2.csv:32>).
- Missing **25**, `set-2` match 25: [rally 24, 11–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Chico_Aura_DWI_WARDOYO_LU_Guang_Zu_PERODUA_Malaysia_Masters_2022_Semifinals/set2.csv:199>) → [rally 26, 13–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Chico_Aura_DWI_WARDOYO_LU_Guang_Zu_PERODUA_Malaysia_Masters_2022_Semifinals/set2.csv:204>).

### 23. An Se Young vs CHEN Yufei — Malaysia Masters 2022, Finals, game 1

Dataset references: `set-2` match **27**. **A:** An Se Young; **B:** CHEN Yufei.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 2 | rally 1: 1–0 | rally 3: 3–0 | Pending |
| 27 | rally 26: 15–11 | rally 28: 16–12 | Pending |

Evidence:

- Missing **2**, `set-2` match 27: [rally 1, 1–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/An_Se_Young_Chen_Yu_Fei_PERODUA_Malaysia_Masters_2022_Finals/set1.csv:2>) → [rally 3, 3–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/An_Se_Young_Chen_Yu_Fei_PERODUA_Malaysia_Masters_2022_Finals/set1.csv:11>).
- Missing **27**, `set-2` match 27: [rally 26, 15–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/An_Se_Young_Chen_Yu_Fei_PERODUA_Malaysia_Masters_2022_Finals/set1.csv:219>) → [rally 28, 16–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/An_Se_Young_Chen_Yu_Fei_PERODUA_Malaysia_Masters_2022_Finals/set1.csv:229>).

### 24. ZHAO Jun Peng vs PRANNOY H. S. — INDONESIA OPEN 2022, Semi-finals, game 2

Dataset references: `set-2` match **29**. **A:** ZHAO Jun Peng; **B:** PRANNOY H. S..

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 3 | rally 2: 2–0 | rally 4: 2–2 | Pending |
| 29 | rally 28: 18–10 | rally 30: 19–11 | Pending |

Evidence:

- Missing **3**, `set-2` match 29: [rally 2, 2–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/EAST_VENTURES_Indonesia_Open_2022_Zhao_Jun_Peng_PRANNOY_H._S._Semifinals/set2.csv:8>) → [rally 4, 2–2](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/EAST_VENTURES_Indonesia_Open_2022_Zhao_Jun_Peng_PRANNOY_H._S._Semifinals/set2.csv:20>).
- Missing **29**, `set-2` match 29: [rally 28, 18–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/EAST_VENTURES_Indonesia_Open_2022_Zhao_Jun_Peng_PRANNOY_H._S._Semifinals/set2.csv:276>) → [rally 30, 19–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/EAST_VENTURES_Indonesia_Open_2022_Zhao_Jun_Peng_PRANNOY_H._S._Semifinals/set2.csv:285>).

### 25. Lakshya SEN vs LEE Zii Jia — YONEX All England Open Badminton Championships 2022, Semi-finals, game 1

Dataset references: `set-2` match **38**. **A:** Lakshya SEN; **B:** LEE Zii Jia.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 14 | rally 13: 7–6 | rally 15: 9–6 | Pending |
| 16 | rally 15: 9–6 | rally 17: 10–7 | Pending |
| 22 | rally 21: 12–9 | rally 23: 13–10 | Pending |
| 27 | rally 26: 14–12 | rally 29: 17–12 | Pending |
| 28 | rally 26: 14–12 | rally 29: 17–12 | Pending |

Evidence:

- Missing **14**, `set-2` match 38: [rally 13, 7–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:173>) → [rally 15, 9–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:184>).
- Missing **16**, `set-2` match 38: [rally 15, 9–6](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:184>) → [rally 17, 10–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:214>).
- Missing **22**, `set-2` match 38: [rally 21, 12–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:296>) → [rally 23, 13–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:312>).
- Missing **27, 28**, `set-2` match 38: [rally 26, 14–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:380>) → [rally 29, 17–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lakshya_SEN_LEE_Zii_Jia_All_England_Open_2022_SemiFinals/set1.csv:399>).

### 26. LEE Zii Jia vs LI Shi Feng — Thailand Open 2022, Finals, game 1

Dataset references: `set-2` match **42**. **A:** LEE Zii Jia; **B:** LI Shi Feng.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 6 | rally 5: 2–3 | rally 7: 4–3 | Pending |
| 25 | rally 24: 12–12 | rally 26: 12–14 | Pending |
| 32 | rally 31: 14–17 | rally 33: 15–18 | Pending |

Evidence:

- Missing **6**, `set-2` match 42: [rally 5, 2–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set1.csv:22>) → [rally 7, 4–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set1.csv:32>).
- Missing **25**, `set-2` match 42: [rally 24, 12–12](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set1.csv:231>) → [rally 26, 12–14](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set1.csv:255>).
- Missing **32**, `set-2` match 42: [rally 31, 14–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set1.csv:291>) → [rally 33, 15–18](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set1.csv:300>).

### 27. LEE Zii Jia vs LI Shi Feng — Thailand Open 2022, Finals, game 2

Dataset references: `set-2` match **42**. **A:** LEE Zii Jia; **B:** LI Shi Feng.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 9 | rally 8: 6–2 | rally 10: 7–3 | Pending |
| 13 | rally 12: 8–4 | rally 14: 9–5 | Pending |
| 22 | rally 21: 13–8 | rally 23: 15–8 | Pending |
| 26 | rally 25: 16–9 | rally 27: 17–10 | Pending |
| 31 | rally 30: 19–11 | rally 32: 21–11 | Pending |

Evidence:

- Missing **9**, `set-2` match 42: [rally 8, 6–2](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:77>) → [rally 10, 7–3](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:79>).
- Missing **13**, `set-2` match 42: [rally 12, 8–4](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:93>) → [rally 14, 9–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:102>).
- Missing **22**, `set-2` match 42: [rally 21, 13–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:195>) → [rally 23, 15–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:201>).
- Missing **26**, `set-2` match 42: [rally 25, 16–9](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:221>) → [rally 27, 17–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:225>).
- Missing **31**, `set-2` match 42: [rally 30, 19–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:241>) → [rally 32, 21–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set2.csv:245>).

### 28. LEE Zii Jia vs LI Shi Feng — Thailand Open 2022, Finals, game 3

Dataset references: `set-2` match **42**. **A:** LEE Zii Jia; **B:** LI Shi Feng.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 2 | rally 1: 0–1 | rally 3: 1–2 | Pending |
| 15 | rally 14: 4–10 | rally 16: 5–11 | Pending |
| 27 | rally 26: 11–15 | rally 28: 12–16 | Pending |
| 32 | rally 31: 14–17 | rally 33: 16–17 | Pending |

Evidence:

- Missing **2**, `set-2` match 42: [rally 1, 0–1](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:2>) → [rally 3, 1–2](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:5>).
- Missing **15**, `set-2` match 42: [rally 14, 4–10](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:146>) → [rally 16, 5–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:158>).
- Missing **27**, `set-2` match 42: [rally 26, 11–15](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:239>) → [rally 28, 12–16](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:250>).
- Missing **32**, `set-2` match 42: [rally 31, 14–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:277>) → [rally 33, 16–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals/set3.csv:291>).

### 29. HE Bingjiao vs CHEN Yufei — Denmark Open 2022, Finals, game 1

Dataset references: `set-2` match **47**. **A:** HE Bingjiao; **B:** CHEN Yufei.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 35 | rally 34: 17–17 | rally 36: 18–18 | Pending |

Evidence:

- Missing **35**, `set-2` match 47: [rally 34, 17–17](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/HE_Bing_Jiao_CHEN_Yu_Fei_Denmark_Open_2022_Final/set1.csv:323>) → [rally 36, 18–18](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/HE_Bing_Jiao_CHEN_Yu_Fei_Denmark_Open_2022_Final/set1.csv:325>).

### 30. HE Bingjiao vs CHEN Yufei — Denmark Open 2022, Finals, game 2

Dataset references: `set-2` match **47**. **A:** HE Bingjiao; **B:** CHEN Yufei.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 8 | rally 7: 2–5 | rally 9: 2–7 | Pending |
| 25 | rally 24: 9–15 | rally 26: 10–16 | Pending |

Evidence:

- Missing **8**, `set-2` match 47: [rally 7, 2–5](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/HE_Bing_Jiao_CHEN_Yu_Fei_Denmark_Open_2022_Final/set2.csv:62>) → [rally 9, 2–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/HE_Bing_Jiao_CHEN_Yu_Fei_Denmark_Open_2022_Final/set2.csv:84>).
- Missing **25**, `set-2` match 47: [rally 24, 9–15](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/HE_Bing_Jiao_CHEN_Yu_Fei_Denmark_Open_2022_Final/set2.csv:238>) → [rally 26, 10–16](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/HE_Bing_Jiao_CHEN_Yu_Fei_Denmark_Open_2022_Final/set2.csv:243>).

### 31. Viktor AXELSEN vs Rasmus GEMKE — French Open 2022, Finals, game 2

Dataset references: `set-2` match **50**. **A:** Viktor AXELSEN; **B:** Rasmus GEMKE.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 20 | rally 19: 11–8 | rally 21: 13–8 | Pending |

Evidence:

- Missing **20**, `set-2` match 50: [rally 19, 11–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Rasmus_GEMKE_French_Open_2022 _Finals/set2.csv:163>) → [rally 21, 13–8](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Rasmus_GEMKE_French_Open_2022 _Finals/set2.csv:195>).

### 32. Viktor AXELSEN vs Kodai NARAOKA — French Open 2022, Semi-finals, game 1

Dataset references: `set-2` match **51**. **A:** Viktor AXELSEN; **B:** Kodai NARAOKA.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 17 | rally 16: 9–7 | rally 18: 11–7 | Pending |

Evidence:

- Missing **17**, `set-2` match 51: [rally 16, 9–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kodai_NARAOKA_YONEX_French_Open_2022_Semifinals/set1.csv:196>) → [rally 18, 11–7](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Viktor_AXELSEN_Kodai_NARAOKA_YONEX_French_Open_2022_Semifinals/set1.csv:205>).

### 33. SHI Yuqi vs Kodai NARAOKA — Denmark Open 2022, Semi-finals, game 1

Dataset references: `set-2` match **52**. **A:** SHI Yuqi; **B:** Kodai NARAOKA.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 28 | rally 27: 16–11 | rally 29: 16–13 | Pending |

Evidence:

- Missing **28**, `set-2` match 52: [rally 27, 16–11](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/SHI_Yu_Qi_Kodai_NARAOKA_Denmark_Open_2022_SemiFinals/set1.csv:276>) → [rally 29, 16–13](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/SHI_Yu_Qi_Kodai_NARAOKA_Denmark_Open_2022_SemiFinals/set1.csv:282>).

### 34. SHI Yuqi vs Kodai NARAOKA — Denmark Open 2022, Semi-finals, game 3

Dataset references: `set-2` match **52**. **A:** SHI Yuqi; **B:** Kodai NARAOKA.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 4 | rally 3: 3–0 | rally 5: 5–0 | Pending |

Evidence:

- Missing **4**, `set-2` match 52: [rally 3, 3–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/SHI_Yu_Qi_Kodai_NARAOKA_Denmark_Open_2022_SemiFinals/set3.csv:50>) → [rally 5, 5–0](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/SHI_Yu_Qi_Kodai_NARAOKA_Denmark_Open_2022_SemiFinals/set3.csv:83>).

### 35. An Se Young vs Pornpawee CHOCHUWONG — Korea Open Badminton Championships 2022, Finals, game 2

Dataset references: `set-2` match **53**. **A:** An Se Young; **B:** Pornpawee CHOCHUWONG.

| Missing rally ID | Previous annotated rally / score | Next annotated rally / score | Status |
| ---: | --- | --- | --- |
| 5 | rally 4: 2–2 | rally 6: 4–2 | Pending |

Evidence:

- Missing **5**, `set-2` match 53: [rally 4, 2–2](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/AN_Se_Young_Pornpawee_CHOCHUWONG_Korea_Open_Badminton_Championships_2022_Finals/set2.csv:35>) → [rally 6, 4–2](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/AN_Se_Young_Pornpawee_CHOCHUWONG_Korea_Open_Badminton_Championships_2022_Finals/set2.csv:54>).

## Possible missing final points — verification required

These files end before a normal completed-game score. **Do not add inferred rally IDs or guessed point outcomes to the internal-omission ledger.** Check the final annotated rally and the remaining video, and compare against the official game result. A wrong end-score label may explain the case without any absent point.

| Review ID | Match and game | Dataset references | Last recorded rally | Last A–B score | Last stroke frame label | Status |
| --- | --- | --- | ---: | --- | --- | --- |
| END-01 | Anders ANTONSEN vs Jonatan CHRISTIE; Indonesia Masters 2020, Quarter-finals; game 2 | `set-1` match **14** | 30 | 10–20 | 77853.0 | Needs video/result verification |
| END-02 | Anthony Sinisuka GINTING vs Viktor AXELSEN; Indonesia Masters 2020, Semi-finals; game 2 | `set-1` match **16** | 31 | 20–11 | 78160.0 | Needs video/result verification |
| END-03 | Anthony Sinisuka GINTING vs Rasmus GEMKE; YONEX THAILAND OPEN 2021, Quarter-finals; game 3 | `set-1` match **24**; `set-2` match **58** | 25 | 20–5 | 129140.0 | Needs video/result verification |
| END-04 | SHI Yuqi vs Kodai NARAOKA; Denmark Open 2022, Semi-finals; game 2 | `set-2` match **52** | 28 | 9–19 | 81335 | Needs video/result verification |
| END-05 | Rasmus GEMKE vs LEE Cheuk Yiu; French Open 2022, Semi-finals; game 1 | `set-2` match **56** | 32 | 19–13 | 46374 | Needs video/result verification |

Source evidence and checks:

- **END-01** — A: Anders ANTONSEN; B: Jonatan CHRISTIE. [set-1 match 14, last rally 30](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals/set2.csv:243>). The last row's `getpoint_player` is `B`; this is an annotation label, not independent confirmation of the game's result. Verify whether the listed last score describes the score before or after that rally and whether further point footage remains.
- **END-02** — A: Anthony Sinisuka GINTING; B: Viktor AXELSEN. [set-1 match 16, last rally 31](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals/set2.csv:185>). The last row's `getpoint_player` is `A`; this is an annotation label, not independent confirmation of the game's result. Verify whether the listed last score describes the score before or after that rally and whether further point footage remains.
- **END-03** — A: Anthony Sinisuka GINTING; B: Rasmus GEMKE. [set-1 match 24, last rally 25](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-1/Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals/set3.csv:230>); [set-2 match 58, last rally 25](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals/set3.csv:230>). The last row's `getpoint_player` is blank; this is an annotation label, not independent confirmation of the game's result. Verify whether the listed last score describes the score before or after that rally and whether further point footage remains.
- **END-04** — A: SHI Yuqi; B: Kodai NARAOKA. [set-2 match 52, last rally 28](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/SHI_Yu_Qi_Kodai_NARAOKA_Denmark_Open_2022_SemiFinals/set2.csv:252>). The last row's `getpoint_player` is `B`; this is an annotation label, not independent confirmation of the game's result. Verify whether the listed last score describes the score before or after that rally and whether further point footage remains.
- **END-05** — A: Rasmus GEMKE; B: LEE Cheuk Yiu. [set-2 match 56, last rally 32](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Rasmus_GEMKE_LEE_Cheuk_Yiu_French_Open_2022_SemiFinals/set1.csv:324>). The last row's `getpoint_player` is `A`; this is an annotation label, not independent confirmation of the game's result. Verify whether the listed last score describes the score before or after that rally and whether further point footage remains.

## Score-label inconsistency — no confirmed point omission

### SCORE-01 — Kunlavut VITIDSARN vs ZHAO Jun Peng, BWF World Championships 2022, Semi-finals, game 1

Dataset reference: `set-2` match **30**. **A:** Kunlavut VITIDSARN; **B:** ZHAO Jun Peng.

- [Rally 34](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Kunlavut_VITIDSARN_ZHAO_Jun_ Peng_BWF_World_Championships_2022_SemiFinals/set1.csv:384>) is labelled **21–13**.
- [Rally 35](<C:/Users/ILLEGEAR/Documents/Post-Game-AR/set-2/Kunlavut_VITIDSARN_ZHAO_Jun_ Peng_BWF_World_Championships_2022_SemiFinals/set1.csv:388>) is labelled **17–18**.
- The consecutive rally IDs have a score change of **-4 for A** and **+5 for B**, which cannot be a one-point progression.
- All rally IDs 1–42 are present in this game. The label 21–13 also appears before the last rally, despite looking like a completed-game score.
- **Status:** needs score-label review. The evidence does not determine which preceding labels are wrong; inspect the surrounding rallies rather than automatically replacing a single row. No missing rally is assigned from this score error.

## How to account for these later

1. Preserve the original CSVs and rally IDs. Keep corrections or recovered records in a separate version until reviewed.
2. For an internal omission, locate the preceding and following annotated rallies using the source links. Review the intervening broadcast footage and record the missing rally's actual outcome, time bounds, stroke labels, and provenance.
3. For an ending-review case, verify the official final score and the footage around the last recorded rally before deciding whether a point is absent or the score is mislabeled.
4. For SCORE-01, review the score sequence surrounding rallies 34–35 and earlier nearby labels. Record the verified before/after values and supporting footage.
5. Update each case's status to `Recovered`, `Annotation error corrected`, or `Unrecoverable`, with a dated evidence note. Apply the same reviewed resolution to every overlapping dataset copy.

## Snapshot provenance

The counts describe the local files inspected for this report, not the published datasets' advertised totals. Copies of the same match are deduplicated by normalized match-folder identity, game number, and rally ID. Differences in letter case, whitespace, and punctuation are ignored for identity matching; duplicate gap and ending scores were checked for agreement.

| Dataset | `match.csv` SHA-256 | Annotation corpus SHA-256 |
| --- | --- | --- |
| `set-1` | `e0a07f6f3dead61fdef27be15acf78e9298d4913bc4718b6fc7d0c9c55ebf6ac` | `441d64dff94f0e1f0223ad44f087462b9e20d2c7b95bdacc508ec040dc2bfbd0` |
| `set-2` | `95260cc5ee102e0ef72ddb54114f13638357e58d68be1e7f3cac06388b7deb34` | `d2f99b03b037fd1583478f763eee5811da59d6782966724f72ee634c998cd698` |

The corpus hash was built in `match.csv` row order, with games in ascending number, from the UTF-8 sequence `relative/path + newline + file SHA-256 + newline` for each declared annotation CSV. These hashes allow a later reviewer to distinguish this snapshot from changed source files.

