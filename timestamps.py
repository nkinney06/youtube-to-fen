import os
import numpy as np
import cv2
from PIL import Image as PImage
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import model_from_json
import scipy.signal
from scipy import stats, signal, ndimage
from pytube import YouTube, Playlist, Channel
import shutil
import chess
import chess.polyglot
import chess.svg
import collections as col
import glob
import re
import sys

def get_content_type(link):
	if '?v=' in link:
		return 'Video'
	elif '/c/' in link or '/user/' in link:
		return 'Channel'
	elif '/playlist?list=' in link:
		return 'Playlist'
	else:
		return 'Unknown'
	return 'Unknown'

def downloadVideo(vidid):
    yt = YouTube('https://www.youtube.com/watch?v=' + vidid)
    
    # download the .mp4 file
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
    video.download()
    
    # make the title alpha-numeric
    orig_title = video.title.replace(".mp4","")
    orig_title = orig_title.replace("'","")
    title = re.sub('[\W_]+', ' ', orig_title, flags=re.UNICODE)
    title = title.strip()
    title = title.replace(" ","_")
    
    # get fps and number of frames
    video = glob.glob("*.mp4")[0]
    vid_cap = cv2.VideoCapture(video)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # get the links
    downloadLink = "https://www.youtube.com/watch?v=" + vidid
    embedLink = downloadLink.replace("watch?v=","embed/")
    
    # return the results
    videoInfo = { "fps": fps, "title": orig_title, "downloadLink": downloadLink, "embedLink": embedLink, "shortName": title, "author": yt.author, "fullTitle": video }
    return videoInfo


def extract_frames(title,every_x_frame):
    vid_cap = cv2.VideoCapture(title + ".mp4")
    frame_cnt = 0
    img_cnt = 0
    while vid_cap.isOpened():
        success,image = vid_cap.read() 
        if not success:
            break
        if (frame_cnt % every_x_frame) == 0:
            if frame_cnt > 0:
                cv2.imwrite("data/"+title+"_frame_"+str(img_cnt)+"_image.jpg", image)
                img_cnt += 1
        frame_cnt += 1
    vid_cap.release()
    cv2.destroyAllWindows()

            
def checkers(bsize,buffs):
    a = np.zeros((8,8))
    a[1::2,::2] = 2
    a[::2,1::2] = 2
    b = np.zeros((8*bsize,8*bsize))
    c = np.ones((bsize,bsize))
    c[buffs:bsize-buffs,buffs:bsize-buffs] = 0
    for i in range(8):
        for j in range(8):
            b[i*bsize:(i+1)*bsize,j*bsize:(j+1)*bsize] = -127 * c * (a[i,j]-1) + 127
    return b


def comb(bsize,buffs): # could be combined with checkers
    a = np.zeros((8,8))
    a[1::2,::2] = 2
    a[::2,1::2] = 2
    b = np.zeros((8*bsize,8*bsize))
    c = np.ones((bsize,bsize))
    c[buffs:bsize-buffs,buffs:bsize-buffs] = 0
    for i in range(8):
        for j in range(8):
            b[i*bsize:(i+1)*bsize,j*bsize:(j+1)*bsize] = -127 * c * (a[i,j]-1) + 127
    rows, cols = np.where(b!=127)
    b[rows,cols] = 1
    rows, cols = np.where(b==127)
    b[rows,cols] = 0
    return b
    
    
def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def gradientx(x):
    gradient_x = make_kernel([[-1.,0., 1.],[-1.,0., 1.],[-1.,0., 1.]])
    return simple_conv(x, gradient_x)


def gradienty(x):
    gradient_y = make_kernel([[-1., -1, -1],[0.,0,0], [1., 1, 1]])
    return simple_conv(x, gradient_y)


def pruneLines(lineset):
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        if np.abs(line - x) < 5: # Within 5 px of the other (allowing for minor image errors)
            cnt += 1
            if cnt == 5:
                end_pos = i+2
                return lineset[start_pos:end_pos]
        else:
            cnt = 0
            x = line
            start_pos = i
    return lineset # Prunes a set of lines to 7 in consistent increasing order (chessboard)


def skeletonize_1d(arr):
    _arr = arr.copy() # create a copy of array to modify without destroying original
    for i in range(_arr.size-1):
        if arr[i] <= _arr[i+1]:
            _arr[i] = 0
    for i in np.arange(_arr.size-1, 0,-1):
        if _arr[i-1] > _arr[i]:
            _arr[i] = 0
    return _arr # return skeletonized 1d array (thin to single value, favor to the right)


def checkMatch(lineset): #Checks whether there exists 7 lines of consistent increasing order
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    for line in linediff:
        if np.abs(line - x) < 5: # Within 5 px of the other (allowing for minor image errors)
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5


def getChessLines(hdx, hdy, hdx_thresh, hdy_thresh): #pixel indices for 7 internal chess lines in x and y
    gausswin = scipy.signal.gaussian(21,4)
    gausswin /= np.sum(gausswin)
    blur_x = np.convolve(hdx > hdx_thresh, gausswin, mode='same')
    blur_y = np.convolve(hdy > hdy_thresh, gausswin, mode='same')
    skel_x = skeletonize_1d(blur_x)
    skel_y = skeletonize_1d(blur_y)
    lines_x = np.where(skel_x)[0] # vertical lines
    lines_y = np.where(skel_y)[0] # horizontal lines
    lines_x = pruneLines(lines_x)
    lines_y = pruneLines(lines_y)
    is_match = len(lines_x) == 7 and len(lines_y) == 7 and checkMatch(lines_x) and checkMatch(lines_y)
    return lines_x, lines_y, is_match


def guessSquareSize(image):
    a = ndimage.gaussian_filter(image, sigma=2)
    A = tf.Variable(a)
    Dx = gradientx(A)
    Dy = gradienty(A)
    Dx_pos = tf.clip_by_value(Dx, 0., 255., name="dx_positive")
    Dx_neg = tf.clip_by_value(Dx, -255., 0., name='dx_negative')
    Dy_pos = tf.clip_by_value(Dy, 0., 255., name="dy_positive")
    Dy_neg = tf.clip_by_value(Dy, -255., 0., name='dy_negative')
    hough_Dx = tf.reduce_sum(Dx_pos, 0) * tf.reduce_sum(-Dx_neg, 0) / (a.shape[0]*a.shape[0])
    hough_Dy = tf.reduce_sum(Dy_pos, 1) * tf.reduce_sum(-Dy_neg, 1) / (a.shape[1]*a.shape[1])
    hough_Dx_thresh = tf.reduce_max(hough_Dx) * 3 / 5
    hough_Dy_thresh = tf.reduce_max(hough_Dy) * 3 / 5
    lines_x, lines_y, is_match = getChessLines(hough_Dx.numpy().flatten(),hough_Dy.numpy().flatten(),hough_Dx_thresh.numpy(),hough_Dy_thresh.numpy())
    lines_x, lines_y, is_match = getChessLines(hough_Dx.numpy().flatten(),hough_Dy.numpy().flatten(),hough_Dx_thresh.numpy(),hough_Dy_thresh.numpy()*.9)
    lines_x_diff = np.diff(lines_x)
    lines_y_diff = np.diff(lines_y)
    mode = stats.mode(np.concatenate((lines_y_diff,lines_x_diff)))[0]
    if len(mode) > 0:
        return int(mode)
    else:
        return -1 
    
    
def trimBoard(board):
    board_bw = np.zeros(board.shape)
    board_bw[np.where(board < np.mean(board))] = 0
    board_bw[np.where(board >= np.mean(board))] = 127
    if int(np.min(board.shape)) < 72:
        return -1
    if int(np.min(board.shape))-7*8 < 1:
        return -1
    squareSize = guessSquareSize(board)
    if (squareSize < 0):
        squareSize = 8
    diffMat = np.zeros((squareSize+1,board.shape[0]-(7)*8,board.shape[1]-(7)*8)) + 9*10**100
    sqar,rows,cols = diffMat.shape
    for i in range(squareSize,squareSize+1):
        Dc = checkers(i,3)
        Cb = comb(i,3)
        for j in range(rows):
            for k in range(cols):
                a = board_bw[j:j+Dc.shape[0],k:k+Dc.shape[1]]
                b = np.zeros(Dc.shape)
                b[0:a.shape[0],0:a.shape[1]] = a
                b = b*Cb+(-127*Cb+127)
                diffMat[i,j,k] = np.mean((Dc-b)**2)
    i,j,k = np.where(diffMat == np.min(diffMat))
    Dc = checkers(i[0],3)
    a = board[j[0]:j[0]+Dc.shape[0],k[0]:k[0]+Dc.shape[1]]
    b = np.zeros(Dc.shape)
    b[0:a.shape[0],0:a.shape[1]] = a
    return b,j[0],k[0],i[0]
       
    
def numpyImage(path,dtype = "float32"):
    orig = PImage.open(path)
    originalPage = np.asarray(orig.convert("L"), dtype=np.float32) if dtype == "float32" else np.array(orig)
    orig.close()
    return originalPage


def display_array(a, fmt='jpeg', rng=[0,1], fName="defaultImage"):
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  PImage.fromarray(a).save(fName,format="PNG")


def imageArrays(images):
    arrays = dict()
    for image in images:
        img = PImage.open(image)
        arrays[image] = np.array(img)
        img.close()
    return arrays


def imageLabels(images):
    pieceCodes = { 'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11,'E':12,'e':12 }
    labels = dict()
    for k in images: # get the labels
        info = [ x.replace(".png","") for x in k.split("_") ]
        labels[k] = pieceCodes[info[-1]] if info[-1] in pieceCodes else int(float(info[-1]))
    return labels


def outputBoards(pageImage,maskImage,prefix):
    orig = numpyImage(pageImage)
    mask = numpyImage(maskImage)
    cols = ['a','b','c','d','e','f','g','h']
    rows = ['8','7','6','5','4','3','2','1']
    labels, nb = ndimage.label(mask)
    for i in range(1,nb+1):
        sl = ndimage.find_objects(labels==i)
        display_array(orig[sl[0]],rng=[0,255],fName=prefix + str(i) + ".png")
        board = numpyImage(prefix + str(i) + ".png")
        squareSizes = [int(x/8) for x in board.shape]
        for x in range(8):
            for y in range(8):
                PImage.fromarray(board[y*squareSizes[0]:(y+1)*squareSizes[0],x*squareSizes[1]:(x+1)*squareSizes[1]]).convert("L").resize([32,32], PImage.ADAPTIVE).save(prefix + str(i) + "_" + cols[x] + rows[y] + ".png")
    return nb

	
def predict_frame(imagePath):

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def parse_image(img_path: str) -> dict:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        return {'image': image }

    @tf.function
    def normalize(input_image: tf.Tensor) -> tuple:
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image

    @tf.function
    def load_image_train(datapoint: dict) -> tuple:
        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
        input_image = normalize(input_image)
        return input_image

    @tf.function
    def load_image_test(datapoint: dict) -> tuple:
        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
        input_image = normalize(input_image)
        return input_image

    IMG_SIZE = 128
    BUFFER_SIZE = 1000
    SEED = 42
    BATCH_SIZE = 1
    pages = tf.data.Dataset.list_files(imagePath)
    pages = pages.map(parse_image)
    dataset = {"train": pages }
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = tf.expand_dims(pred_mask, axis=-1)
        return pred_mask

    json_file = open('frameModel.json', 'r') # load json and create model
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('frameModel.h5') # load weights into new model
    for image in dataset['train'].take(1):
        sample_image = image
    pred_mask = create_mask(loaded_model.predict(sample_image[0][tf.newaxis, ...]))
    display_list = [sample_image[0],pred_mask[0]]
    titles = ['Input Page', 'Predicted Mask']
    return pred_mask[0]


def useNetwork(images,network,img_rows=32,img_cols=32):
    json_file = open('./' + network + '.json', 'r') # load json and create model
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('./' + network + '.h5') # load weights into new model
    x_input, imName = ([] for i in range(2))
    for k in images: # convert images to numpy
        x_input.append(images[k])
        imName.append(k)
    x_input = np.array(x_input)
    if K.image_data_format() == 'channels_first':
        x_input = x_input.reshape(x_input.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_input = x_input.reshape(x_input.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_input = x_input.astype('float32') / 255
    predictions = dict()
    confidences = dict()
    for obs in range(len(x_input)):
        input = x_input[None,obs]
        preds = loaded_model.predict(input,verbose=0)
        predictions[imName[obs]] = np.argmax(preds[0])
        confidences[imName[obs][-6:-4]] = max(preds[0])
    return predictions,confidences

        
def imageArrays(images):
    arrays = dict()
    for image in images:
        img = PImage.open(image)
        arrays[image] = np.array(img)
        img.close()
    return arrays


def reverseVideoMask(title,frameNum,pred_mask):
        im = numpyImage("data/" + title + "_frame_" + str(frameNum) + "_image.jpg")
        prediction = tf.keras.preprocessing.image.array_to_img(tf.image.resize(pred_mask, im.shape))
        tf.keras.utils.save_img("data/" + title + "_frame_" + str(frameNum) + "_mask.jpg",tf.image.resize(pred_mask, im.shape))
        mask = numpyImage("data/" + title + "_frame_" + str(frameNum) + "_mask.jpg")
        orig = numpyImage("data/" + title + "_frame_" + str(frameNum) + "_image.jpg")   
        labels, nb = ndimage.label(mask)
        boardsDetected = []
        endMask = np.zeros(orig.shape)
        for i in range(1,nb+1):
            sl = ndimage.find_objects(labels==i)
            length,width = orig[sl[0]].shape
            if ( ( length > 120 ) & ( width > 120 ) ):
                trimmed,ox,oy,oz = trimBoard(orig[sl[0]])
                boardsDetected.append(trimmed)
                display_array(orig[sl[0]],rng=[0,255],fName="data/" + title + "_frame_" + str(frameNum) + "_board_" + str(len(boardsDetected)) + ".png")
                endMask[sl[0][0].start + ox:sl[0][0].start + ox + 8*oz,sl[0][1].start + oy:sl[0][1].start + oy + 8*oz] = 255      
        display_array(endMask,rng=[0,255],fName="data/" + title + "_frame_" + str(frameNum) + "_mask.jpg")

        
def predictFrames(title,frameNum,downloadLink,embedLink,fps,channel):
    Frame = col.namedtuple('Frame', ['title', 'time', 'link', 'zobristWhite', 'fenStringWhite', 'turn', 'fps', 'downloadLink', 'embedLink', 'author'])
    frames = glob.glob("data/" + title + "_frame_*_image.jpg")
    pieces = ['P','N','B','R','Q','K','p','n','b','r','q','k','E','E']
    for i in range(len(frames)):
        prefix = "data/" + title + "_frame_" + str(i) + "_board_"
        numBoards = outputBoards("data/"+title+"_frame_"+str(i)+"_image.jpg","data/"+title+"_frame_"+str(frameNum)+"_mask.jpg",prefix)
        for boardNum in range(1,numBoards+1):
            images = glob.glob("data/" + title + "_frame_" + str(i) + "_board_" + str(boardNum) + "_*.png")
            predictions,confidence = useNetwork(imageArrays(images),"pieceModelVideo")
            guesses = [[x,pieces[y]] for (x,y) in predictions.items() if pieces[y] != 'E']
            b = chess.Board()
            b.clear()
            flip = False
            [ b.set_piece_at(chess.SQUARE_NAMES.index(x[0][-6:-4]),chess.Piece.from_symbol(x[1])) for x in guesses if x[1] != 'e' ]
            bflip = chess.Board() # flip the board if necessary
            bflip.clear_board()
            if ( len(chess.SquareSet(b.occupied_co[0])) > 0 ):
                if ( len(chess.SquareSet(b.occupied_co[1])) > 0 ):
                    if ( sum(chess.SquareSet(b.occupied_co[0]))/len(chess.SquareSet(b.occupied_co[0])) < sum(chess.SquareSet(b.occupied_co[1]))/len(chess.SquareSet(b.occupied_co[1])) ):
                        for square in chess.SquareSet(b.occupied_co[0]):
                            bflip.set_piece_at(chess.SQUARES[63-square],b.piece_at(square))
                        for square in chess.SquareSet(b.occupied_co[1]):
                            bflip.set_piece_at(chess.SQUARES[63-square],b.piece_at(square))
                        b = bflip
                        flip = True
            fenStringWhite = b.fen()
            zobristWhite = chess.polyglot.zobrist_hash(b)
            timepoint = i
            F = Frame(title,timepoint,downloadLink + "&t=" + str(timepoint),str(zobristWhite).rstrip(),fenStringWhite.strip(),1,fps,downloadLink,embedLink,title)
            print(title+"\t"+F.link+"\t"+F.zobristWhite+"\t"+F.fenStringWhite)
                
                
def outputFens(vidid):
    videoInfo = downloadVideo(vidid)
    title = videoInfo['shortName']
    channel = videoInfo['author']
    shutil.copy(glob.glob("*.mp4")[0], title + ".mp4")
    extract_frames(title,int(videoInfo['fps']))
    extractedframes = glob.glob("data/" + title + "_frame_*_image.jpg")
    pred_mask = predict_frame("data/" + title + "_frame_" + str(int(len(extractedframes)/2)) + "_image.jpg")
    reverseVideoMask(title,int(len(extractedframes)/2),pred_mask)
    predictFrames(title,int(len(extractedframes)/2),videoInfo['downloadLink'],videoInfo['embedLink'],videoInfo['fps'],channel)
    for myfile in glob.glob("*.mp4"):
        os.remove(myfile)
    for myfile in glob.glob("data/*"):
        os.remove(myfile)


def print_usage():
    print("usage:")
    print("python timestamps.py https://www.youtube.com/watch?v=your_video_id")
    print("example:")
    print("python timestamps.py https://www.youtube.com/watch?v=_j1dNtN6zow")
    exit()


############################################################################################################
## main programming section
############################################################################################################
if len(sys.argv) < 2:
    print_usage()
content = sys.argv[1]
content_type = get_content_type(content)
if content_type == "Video":
    videoID = content.replace('https://www.youtube.com/watch?v=','')
    outputFens(videoID)
    exit()
else:
    print_usage()





		
		
		
		
		
		
