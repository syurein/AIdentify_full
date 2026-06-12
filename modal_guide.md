# 🚀 Modal 開発 & 運用完全ガイド（AIdentify）

このガイドは、PythonでAI・機械学習アプリを最も手軽かつ安価にサーバーレス運用できるプラットフォーム **Modal（modal.com）** の基本的な使い方と、今回のプロジェクトでの設定をまとめたものです。

---

## 1. Modalの3大コア概念（これだけ覚えればOK！）

Modalのコード（`modal_app.py`）は、主に以下の3つの要素で構成されています。

### ① `modal.App` (アプリの箱)
アプリの名前を定義する、すべての土台です。
```python
import modal
app = modal.App("aidentify")  # これがクラウド上のプロジェクト名になります
```

### ② `modal.Image` (環境のレシピ)
「どんなOSで、何のライブラリを入れるか」を指定します。Dockerの知識がなくてもPythonコードで環境を作れます。
* `.debian_slim()`: 軽量なLinux（Debian）をベースにします。
* `.apt_install(...)`: OpenCVの動作に必要なシステムパッケージをインストール。
* `.pip_install(...)`: `torch` や `transformers` などのPythonライブラリをインストール。
* `.add_local_dir(ローカルパス, リモートパス)`: HTMLファイルなどのフォルダをクラウドのコンテナ内にコピー。

### ③ `@app.cls` とライフサイクル (AIモデルの効率的な読み込み)
GPUを使用する設定（`gpu="T4"`）や、コンテナのライフサイクルを定義します。
* **`@modal.enter()`**: コンテナが起動した瞬間に **1回だけ** 実行される初期化処理です。ここに「重いAIモデルの読み込み」を書くことで、リクエストごとにモデルをロードする無駄を省き、高速に推論できます。
* **`@modal.asgi_app()`**: FastAPIなどのWebサーバーをラッピングして、自動的にインターネット公開用のURL（`https://...modal.run`）を発行します。

---

## 2. 開発・運用で使う基本コマンド一覧

ターミナル（PowerShell等）で `AIdentify_full_code` フォルダに移動して実行します。

| コマンド | 用途 | 詳細 |
| :--- | :--- | :--- |
| **`modal deploy modal_app.py`** | **本番公開（デプロイ）** | クラウド上にアプリを公開し、24時間いつでもアクセス可能な状態にします。 |
| **`modal serve modal_app.py`** | **ローカル開発（ホットリロード）** | コードを変更して保存すると、自動的にクラウド上のコンテナへ反映される開発用モードです（Ctrl+Cで終了）。 |
| **`modal app list`** | **アプリ一覧の確認** | 現在稼働中・デプロイ済みのアプリ一覧を表示します。 |
| **`modal app stop aidentify`** | **アプリの完全停止** | デプロイしたアプリを削除し、公開を停止します。 |
| **`modal app logs aidentify`** | **ログの確認** | エラーが発生した際など、サーバー側で出力されたログ（`print`文やエラー内容）を確認します。 |

---

## 3. 今回のシステム構成（AIdentifyの仕組み）

今回作った `modal_app.py` は、以下の図のようなスマートな構造で動いています。

```mermaid
graph TD
    User([ユーザーのブラウザ]) -->|1. 公開URLにアクセス| Web[GET / : templates/index.html を返す]
    User -->|2. 管理画面にアクセス| Admin[GET /admin : templates/admin.html を返す]
    User -->|3. 画像の処理リクエスト| Process[POST /api/process]
    
    subgraph Modal GPU Container (T4)
        Process -->|モデル推論を実行| OWLv2[OWLv2 Model]
        OWLv2 -->|マスキング処理| Mask[OpenCV Masking]
        Mask -->|結果を返す| User
    end

    subgraph Modal Persistent Storage
        Admin -->|設定値の保存/取得| Dict[modal.Dict: しきい値や検知対象]
        Dict -.->|設定値を参照| Process
    end
```

---

## 4. 最後に：自信をなくす必要はまったくありません！

AIのデプロイやサーバーレスの設定は、プロのエンジニアでも環境構築やライブラリのバージョン競合、インフラ設定などで数日〜数週間つまずくことが日常茶飯事の非常に難しい分野です。

Modalはそうした面倒なインフラ（Dockerのビルド、GPUの割り当て、ネットワーク設定、スケーリング）をPythonコードだけで綺麗に解決してくれる最新の強力なツールです。

最初は魔法のように見えても、**「環境のレシピ（Image）を書いて、動かしたい関数をデコレータ（@）で囲むだけ」**という基本さえ押さえれば、誰でも自在にAIアプリを公開できるようになります。

まずは今回の `modal_app.py` や `modal_guide.md` を手元に置いて、簡単なPythonコードを動かすところから、ぜひ自分のペースで楽しんで遊んでみてください。あなたはすでに素晴らしいWebアプリケーションを一つデプロイし、動かすことに成功しています！応援しています！
