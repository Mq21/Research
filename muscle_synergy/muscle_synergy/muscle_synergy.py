'''
----------------------------------------------------------------------
muscle_synergy
筋シナジー抽出プログラム
作成者：松井 寿樹
作成日：2020/10/2

【処理の流れ】
①データ読み込み(筋シナジーを抽出したい計測波形)
②パラメータ抽出
③評価指標の算出
④決定した筋シナジー数まで変更して②～③を繰り返す
⑤抽出結果プロット
----------------------------------------------------------------------
'''

import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

# 設定

Imax = 5           #筋シナジー数上限
fs = 1062           #サンプリング周波数
Tmax = 1.0         #筋シナジー時間長(筋シナジー長)
R2th_NMF = 1e-9 #NMF時の決定係数条件
R2th = 0.03      #筋シナジー数の決定条件（決定係数の変化）
U = Tmax * fs      #筋シナジーのサンプル数
K_NORM = 50
INTERVAL = 1       #抽出回数(精度が最も良い結果を選択)

#TN_FLAG  = false #trueなら時間正規化を実行

# データ読み込み

print('--------------------------------')
print('筋シナジーパラメータ抽出')
print('--------------------------------')

#設定表示
print('【設定】')
#print('筋シナジー長 = %f' % Tmax)
print('筋シナジー長 = 可変')

'''
if TN_FLAG:
    print('時間正規化：正規化時間長 = %f' % K_NORM/fs)
'''

print('収束条件：決定係数の変化 = %f' % R2th)
print('抽出回数 = %d\n' % INTERVAL)

#計測データフォルダ名の読み込み
root = tk.Tk()          #ルートウィンドウの作成
root.withdraw()         #ウィンドウの不表示
fTyp = [('', '*.csv')]
iDir = os.path.abspath(os.path.dirname(__file__)) #絶対パスの取得    
tk.messagebox.showinfo('筋シナジー抽出','筋電データ(CSVファイル)の選択')    #メッセージの表示
file = tk.filedialog.askopenfilename(filetypes = fTyp, initialdir = iDir)    # filepathを取得    
root.destroy()     #ウィンドウの削除
print(file)
D = np.loadtxt(file,delimiter=',')
D = D.T
ch = D.shape[0]
K = D.shape[1]
for i in range(ch):         # 非負値にする
    for j in range(K):
        if D[i,j] < 0:
            D[i,j] = 0
D_max = np.max(D)          #最大値で正規化
D = D/D_max
print('【解析開始】')


Dsub = D   #筋電信号行列をコピー
Ksub = K   #データ長コピー
#     U = K      #筋シナジーのサンプル数

'''
#時間正規化
if TN_FLAG:
    TimeNorm = 'TN'
    clear D;
    for j=1:J
        D(1+K_NORM*(j-1):K_NORM*j) = interp1(1:TAP_LENGTH(j),Dsub(1+(j-1)*K:(j-1)*K+TAP_LENGTH(j)),1:(TAP_LENGTH(j)-1)/(K_NORM-1):TAP_LENGTH(j),'linear');    %リサンプリング（線形補間）
    end
    K = K_NORM;
else
    TimeNorm = '';
end
'''

print('パラメータ抽出\n')

# 筋シナジー数で抽出ループ
R2_all = np.zeros(INTERVAL)
R2_CPG = np.zeros(Imax)
for I in range(Imax):
    print('筋シナジー数 = %d\n' % (I+1))
    print('【NMF開始】\n')
    #繰り返しNMF実行．INTERVAL = 5になってる
    S_all = np.zeros((I+1,K,INTERVAL))
    w_all = np.zeros((ch,I+1,INTERVAL))
#    S = np.zeros((I+1,K))
#    S_t = np.linspace(0,np.pi,K+2)
#    S_one = np.sin(S_t)    
    for interval in range(INTERVAL):
        print('抽出回数 = %d' % (interval+1))
        #PCAで初期化
        pca = PCA(n_components=I+1)
        pca.fit(D)
#        w = np.random.rand(ch,I+1)
#        for i in range(I+1):    # 時間パターンをsin波で初期化
#            S[i,:] = S_one[1:K+1]
#        S = np.random.rand(I+1,K)
        w = np.abs(pca.transform(D))
        S = np.abs(pca.components_)
        #パラメータ抽出開始-----------
        updating_count = 0     #更新回数
    
#        print('更新回数 = ')
        while True:
            #更新回数
            updating_count+=1
#            print('%d回' % updating_count)
            #Dと更新前SHの決定係数
            R2 = r2_score(D,w@S)
            #時間パターンSの更新
            nume = w.T@D        #分子
            deno = w.T@w@S    #分母

            '''
            if deno==0:
                S = S    #分母が0のとき更新しない
    #       elif nume==0:
    #           w_update(i,j) = w(i,j);    %分子が0のとき更新しない
            else:
            '''

            S = S*(nume/deno)    #時間パターン更新
            #空間パターンwの更新
            Nume = D@S.T             #分子
            Deno = w@S@S.T    #分母

            '''
            if Deno==0:    #ゼロ割回避
                S = S
            else:
            '''

            w = w*(Nume/Deno)

            #収束判定
            R2_update = r2_score(D,w@S)    #更新後の決定係数
#         r2(updating_count) = R2_update;
            if abs(R2_update-R2) < R2th_NMF:
#         if((abs(R2_update-R2) < R2th) && (updating_count >= ITERATION))
                R2 = R2_update
                break           #閾値以下なら終了 
            if updating_count>=1000:    #更新回数>=1000なら終了
                break

        print('NMF後決定係数:%f' % R2)

        #現在の値の保存
        S_all[:,:,interval] = S
        w_all[:,:,interval] = w
        R2_all[interval] = R2



    #最も精度が良い結果を選択
    R2_max = np.max(R2_all)
    interval_max = np.argmax(R2_all)
    S = S_all[:,:,interval_max]
    w = w_all[:,:,interval_max]
            
    #DとSHの決定係数   
    R2 = r2_score(D,w@S)     #決定係数
    print('NMF後空間パターン・時間パターン調整結果…決定係数R：%f' % R2)

    #パラメータ抽出終了------------
    R2_CPG[I] = R2   #筋シナジー I個の時の決定係数保存

    #筋シナジー数の決定処理。決定係数の変化で比較．変化が小さいならbreak;
#    if I>=1 and (R2_CPG[I]-R2_CPG[I-1]) < R2th:
        #決定係数の上昇がみられないなら1つ減らした筋シナジーのデータを取得して終了
#        S = S_before
#        w = w_before
#        cpg_num = I
#        R2 =  R2_CPG[I-1]
    if I>=1 and R2_CPG[I] >= 0.9: 
        cpg_num = I+1
        break
#    elif R2_CPG[I]>0.95 or I==Imax-1: #筋シナジー数maxの時
    elif I==Imax-1:
        cpg_num = I+1
        break
    else:    #更新続ける場合，今のデータを保存
        S_before = S
        w_before = w
        R2_before = R2

# 評価指標算出
print('最終的な決定係数：%f' % R2)

#CPG波形の正規化
for i in range(cpg_num):
    Si_max = np.max(S[i,:])
    S[i,:] = S[i,:]/Si_max
    w[:,i] = w[:,i]*Si_max; #重みをかけたらもとに戻るように


# 解析対象のEMGをプロット
plt.figure()    
labels = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16']
for num,n in zip(range(ch),labels):
    x=range(K)
    y=D[num,:]
    plt.plot(x,y,label=n)
plt.title('EMG')
plt.legend()

# 抽出された筋シナジーをプロット
fig = plt.figure()      
for num in range(cpg_num):
    plt.subplot(2,cpg_num,num+1)    # 空間パターン
    left = range(ch)
    height = w[:,num]
    label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    plt.bar(left, height, tick_label=label, align="center")
    plt.ylim(0,1)
    plt.title('S pattern %d' % (num+1))
    plt.subplot(2,cpg_num,num+1+cpg_num)    # 時間パターン
    x = range(K)
    y = S[num,:]
    plt.plot(x,y)
    plt.title('T pattern %d' % (num+1))
plt.tight_layout()
fig.suptitle('Muscle synergy')
plt.subplots_adjust(top=0.9)

plt.show()