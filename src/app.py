# 必要なモジュールのインポート
import os
import torch
# from animal import transform, Net #animal.pyから前処理とネットワークの定義を読み込み

from flask import Flask, render_template, request, redirect
import io
from PIL import Image
import base64
from ultralytics import YOLO
from glob import glob

# 学習済みモデルをもとに推論する
def predict(img):
    # # ネットワークの準備
    # net = Net().cpu().eval()

    # # py ファイルのディレクトリを取得
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # py ファイルのディレクトリに基づく相対パスでファイルを指定
    # file_path = os.path.join(script_dir, 'dog_cat_weights_1CNN.pt')
    
    # # 学習済みモデルの重み（dog_cat.pt）を読み込み
    # net.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    # #　データの前処理
    # img = transform(img)
    # # (batch_size, channel, height, width) の形式に変換
    # img =img.unsqueeze(0) # 1次元増やす
    # #　推論
    # y = torch.argmax(net(img), dim=1).cpu().detach().numpy()

    return y

def yolo_predict(img_path):
    model_path = "best.pt"
    model = YOLO(model_path)

    model.predict(img_path, save=True)

    return


#　推論したラベルから犬か猫かを返す
def getName(label):
    if label==0:
        return "猫"
    elif label==1:
        return "犬"

# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    print('/ディレクトリへのアクセス')
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        print("POST")
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            print('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            print(file)
            print(file.filename)
            yolo_predict(file.filename)

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            path_list = sorted(glob('runs/detect/*'))
            file_path = path_list[-1]
            print(file_path)
            while True:
                if file.filename in os.listdir(file_path):
                    print('predected!')
                    break
            image_path = f'runs/detect/predict/{file.filename}'
            image = Image.open(image_path).convert('RGB')

            # 画像の大きさを調整する
            # 講義資料にはなく、追加しています！
            new_width = 500
            new_height = 500
            image = image.resize((new_width, new_height))

            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            #　HTML側のsrcの記述に合わせるために付帯情報付与する
            base64_data = "data:image/png;base64,{}".format(base64_str)

            # 入力された画像に対して推論
            # pred = predict(image)
            # animalName_ = getName(pred)
            return render_template('result.html', img=base64_data)

    # GET 　メソッドの定義
    elif request.method == 'GET':
        print("GET")
        return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == "__main__":
    app.run(port=5000, debug=True)