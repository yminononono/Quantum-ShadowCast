# ShadowCast — Josephson Junction 斜め蒸着シミュレーター

斜め蒸着（shadow / oblique evaporation）による **ジョセフソン接合（Josephson junction）** の
作製プロセスを 3D でシミュレーションする Streamlit アプリです。
**Dolan bridge** 方式と **Manhattan** 方式の両方に対応し、レジスト形状・蒸着角度から
接合の重なり面積・臨界電流・インダクタンス・ジョセフソンエネルギーまでを計算します。

物理計算の中心は 3D ボクセル・レイキャスト・エンジン（`deposition3d.py`）で、
傾いた蒸着ビームをボクセルグリッドに ray-trace して各金属膜の堆積領域を求め、
1 回目と 2 回目の金属膜が酸化膜を介して重なる領域を接合として抽出します。
この 3D エンジンが「真値（source of truth）」であり、画面表示・判定はこれに基づきます。

---

## 主な機能

- **2 つの作製モード**
  - **Dolan bridge**: 単軸傾斜（同一 φ・反転 θ）でブリッジ下に蒸着を回り込ませる方式
  - **Manhattan**: 2 本の蒸着ビーム（θ₁/φ₁、θ₂/φ₂ を独立に設定）を交差させる方式
- **3D 斜め蒸着エンジン**: ボクセル・レイキャストで金属膜・酸化膜・アンダーカットを再現
- **任意形状の接合面積**: 接合を「正方形」と仮定せず、1 回目膜 ∩ 2 回目膜の
  実際の重なり領域（長方形でなくても）をセル数から面積として算出
- **電気特性の計算**: 接合面積から
  - 臨界電流 Ic（Ambegaokar–Baratoff, jc = 10 kA/cm²）
  - ジョセフソンインダクタンス L_J = ħ / (2e·Ic)
  - ジョセフソンエネルギー E_J = (Φ₀/2π)·Ic（E_J/h [GHz]、E_J/k_B [K] でも表示）
- **6 つの可視化タブ**
  1. **📐 Cross-section** — 任意角度・オフセットで回転できる断面図（蒸着ビーム矢印つき）
  2. **🗺️ Top View** — 上面図（金属膜・シャドウ・アンダーカット・接合領域）
  3. **🔄 φ Junction View** — 接合まわりの拡大上面図
  4. **🔍 Break Check** — open/short 判定と電気特性メトリクス
  5. **📈 Parameter Scan** — パラメータ掃引（下記）
  6. **📊 Junction Area** — 全パラメータ要約と結果のエクスポート
- **Parameter Scan（パラメータ掃引）**
  - **1D / 2D** 掃引（2D はヒートマップ）
  - 掃引する **値の範囲と分割数** を各変数ごとに指定可能
  - 掃引の **ボクセル密度（精度）** はサイドバーと同じ 5 段階から選択
  - 出力は接合面積・Ic・L_J・E_J/h・E_J/k_B を **すべて縦に並べて** 表示
  - Manhattan では各蒸着の **θ₁ / φ₁ / θ₂ / φ₂** も掃引対象
- **パラメータの保存・読み込み**: 設定を JSON で保存／復元、デフォルトへのリセットボタン
- **GDS 読み込み**（任意）: GDSII レイアウトファイルの読み込みに対応（`gdstk` が必要）

---

## 動作環境

- **Python 3.10 以上**（開発・検証は 3.11 で実施）
- 主要パッケージ: `streamlit`, `numpy`, `matplotlib`
- GDS 読み込みを使う場合のみ: `gdstk`

依存パッケージは最小限です（`scipy` / `rtree` などは不要）。

---

## インストール

```bash
# 1. このディレクトリに移動
cd ShadowCast_v6_modify

# 2.（推奨）仮想環境を作成
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 依存パッケージをインストール
pip install -r requirements.txt
```

`requirements.txt` の最小構成だけで十分動きます:

```
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
gdstk>=0.9.0          # GDS 読み込みを使う場合のみ
```

---

## 起動方法

```bash
streamlit run app.py
```

ブラウザが自動で開きます（開かない場合はターミナルに表示される
`http://localhost:8501` を開いてください）。
左サイドバーでパラメータを設定すると、各タブの図と計算結果がリアルタイムに更新されます。

---

## 使い方

1. **左サイドバーでモードを選択**（Dolan bridge / Manhattan）
2. **レジスト・蒸着パラメータを設定**
   - 共通: PMMA 厚、MMA 厚、アンダーカット、Evaporation 1（θ₁ / φ₁ / 金属厚 d₁）
   - Dolan: Evaporation 2（θ₂ / φ₂ / d₂）、ブリッジ寸法
   - Manhattan: Evaporation 2（θ₂ / φ₂ / d₂）、x/y アーム開口幅
3. **Ray-scan resolution** でボクセル密度（精度 vs 速度）を選択
4. 各タブで断面・上面・接合を確認。**Break Check** で open/short 判定
5. **Parameter Scan** タブで範囲・分割数・精度を決めて **▶ Run scan**
6. **Junction Area** タブで結果を JSON エクスポート

> 精度を上げる（`Maximum (slowest)` など）と計算が重くなります。まずは
> `Standard (fast)` で当たりをつけ、必要に応じて細かくしてください。
> 2D スキャンは「分割数 × 分割数」回エンジンを実行するため、点数に注意してください。

---

## 主なパラメータ

| パラメータ | 説明 |
|---|---|
| `t_pmma` | PMMA（上層レジスト）厚 [nm] |
| `t_mma` | MMA（下層レジスト）厚 = ブリッジ下の空隙高さ [nm] |
| `undercut` | MMA の片側アンダーカット量 [nm] |
| θ₁ / φ₁ / d₁ | 1 回目蒸着の極角 / 方位角 / 金属厚 |
| θ₂ / φ₂ / d₂ | 2 回目蒸着の極角 / 方位角 / 金属厚 |
| `bridge_len` / `bridge_w` | （Dolan）ブリッジの長さ / 幅 [nm] |
| `manhattan_wx` / `manhattan_wy` | （Manhattan）x / y アーム開口幅 [nm] |

蒸着ビーム方向は `beam = (sinθcosφ, sinθsinφ, −cosθ)`（θ は法線からの傾き）。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `app.py` | Streamlit UI 本体（タブ・サイドバー・スキャン・エクスポート） |
| `deposition3d.py` | **3D ボクセル蒸着エンジン**（`simulate` / `junction_footprint`）= 真値 |
| `voxel_view.py` | 3D 結果の断面・上面レンダリング |
| `junction_area.py` | 解析的な接合面積モデル（補助・概算用） |
| `process_engine.py` | `ProcessParams` データクラスと幾何ヘルパ |
| `cross_section.py` / `phi_cross_section.py` / `top_view.py` | 2D 作図ユーティリティ |
| `manhattan_check.py` | Manhattan の open/short チェック |
| `gds_parser.py` / `generate_sample_gds.py` | GDS 読み込み・サンプル生成（任意） |
| `requirements.txt` | 依存パッケージ |

---

## 計算の前提・注意

- 臨界電流は `Ic[µA] = (接合面積[nm²]) × 1e-4` （jc = 10 kA/cm²、Al・~4 K の目安）。
  材料・条件で変わるため、絶対値は概算として扱ってください。
- ボクセル分解能が粗いと面積・open/short 判定が量子化誤差を含みます。
  境界付近の条件では分解能を上げて確認してください。
- 解析モデル（`junction_area.py`）は概算・トレンド把握用で、最終判定は
  3D エンジン（`deposition3d.py`）に基づきます。
