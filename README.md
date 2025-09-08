# MyBatis Mapper Analyzer

MyBatis の Mapper（XML / Java）を解析し、SQL の複雑度や依存関係、テーブル利用状況を Markdown レポートや CSV/画像として出力するツールです。

## 概要
- Mapper ファイルを走査して SQL 文を抽出・解析
- SQL ごとの複雑度、行数、テーブル、パラメータなどを CSV 出力
- 依存関係グラフ（隣接リスト）と循環参照の検出
- Markdown レポート（summary、ファイル一覧、依存関係、クラスタ図）を生成

## 必要条件
- Python 3.8+
- 推奨（オプション）
  - sqlglot（より正確な SQL 抽出 / エイリアス解析）
  - matplotlib（複雑度 vs LOC の散布図生成）
  - graphviz（依存関係の画像出力）およびシステム側の graphviz 実行環境

インストール例:
```
pip install sqlglot matplotlib graphviz
# system: brew install graphviz などが必要
```

## インストール
リポジトリをクローンしてそのまま実行できます。追加の Python モジュールは上記を参照して導入してください。

## 使い方
基本的な実行例:
```
python mybatis_analyzer.py <解析対象ディレクトリ> -o output/mapper_analysis.md --csv output/mapper_sql_metrics.csv --adjacency output/mapper_adjacency.md --graph output/mapper_graph.png
```

コマンドライン引数（主なもの）
- path: 解析対象のルートディレクトリ
- -o, --output: Markdown レポート出力先（デフォルト: output/mapper_analysis.md）
- --csv: SQL ごとのメトリクス CSV（デフォルト: output/mapper_sql_metrics.csv。空文字で無効化）
- --adjacency: 隣接リスト Markdown（空文字で無効化）
- --graph: 依存関係グラフ画像（空文字で無効化）
- --table-map: テーブル略称→正式名の JSON（任意）

## 出力
- Markdown レポート（指定した -o）
  - サマリー、ファイル統計、SQL 統計、依存関係、テーブル一覧、Mapper 詳細、クラスタ図の埋め込み
- CSV（--csv 指定）
  - namespace, file, statement_id, type, line_count, element_count, complexity, has_dynamic, sql_hash, tables, param_count, cluster, ...
  - 同一ディレクトリにクラスタ画像（*_complexity_loc.png）とクラスタ集計 JSON（*_clusters.json）を生成
- 隣接リスト Markdown / JSON（--adjacency 指定）
- Graphviz PNG（--graph 指定。graphviz ライブラリとシステム実行環境が必要）

## 補足
- xml の include 展開や resultMap の extends を考慮して依存関係を解析します。
- sqlglot があるとテーブル抽出やエイリアス解決が改善されます。無い場合は正規表現ベースのフォールバック処理が動作します。
- Graphviz 出力には Python graphviz パッケージとシステム graphviz（dot コマンド）が必要です。

## 開発 / 貢献
- バグ報告や改善提案は Issue を作成してください。
