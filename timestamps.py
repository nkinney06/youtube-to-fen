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


def print_usage():
    print("usage:")
    print("python timestamps.py https://www.youtube.com/watch?v=your_video_id")
    print("example:")
    print("python timestamps.py https://www.youtube.com/watch?v=_j1dNtN6zow")
    exit()


def downloadVideo(link):
    if '?v=' not in link:
        print_usage()
        
    yt = YouTube(link)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
    video.download()
    
    # make the title alpha-numeric
    alphaNumericTitle = yt.title.replace("'","")
    alphaNumericTitle = re.sub('[\W_]+', ' ', alphaNumericTitle, flags=re.UNICODE)
    alphaNumericTitle = alphaNumericTitle.strip()
    alphaNumericTitle = alphaNumericTitle.replace(" ","_")

    # not sure this is necessary
    mp4 = glob.glob("*.mp4")[0]
    shutil.copy(mp4, alphaNumericTitle + ".mp4")
    
    # get the fps
    vid = cv2.VideoCapture(mp4)
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    # get the frames
    framen = 0  # frame number
    frames = [] # list of frames
    while vid.isOpened():
        success,image = vid.read() 
        if not success:
            break
        if (framen % fps) == 0:
            if framen > 0:
                frames.append(image)
        framen += 1
    vid.release()
    cv2.destroyAllWindows()
    
    # return dictionary including list of the video frames
    videoInfo = { "fps": fps, "title": alphaNumericTitle, "author": yt.author, "frames":frames }
    return videoInfo


def normalize_frame(frame):
    frame = tf.keras.utils.array_to_img(frame)
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize(frame, (128, 128))
    frame = tf.cast(frame, tf.float32) / 255.0
    return frame


def predict_frame(frame):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = tf.expand_dims(pred_mask, axis=-1)
        return pred_mask
    
    frame = normalize_frame(frame)
    json_file = open('frameModel.json', 'r') # load json and create model
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('frameModel.h5') # load weights into new model
    pred_mask = create_mask(loaded_model.predict(frame[tf.newaxis, ...]))
    return pred_mask[0]


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


def reverseVideoMask(frame,pred_mask):
    frame = PImage.fromarray(frame)
    frame = np.asarray(frame.convert("L"), dtype=np.float32)
    prediction = tf.keras.preprocessing.image.array_to_img(tf.image.resize(pred_mask, frame.shape))
    labels, nb = ndimage.label(prediction)
    boardsDetected = []
    endMask = np.zeros(frame.shape)
    for i in range(1,nb+1):
        sl = ndimage.find_objects(labels==i)
        length,width = frame[sl[0]].shape
        if ( ( length > 120 ) & ( width > 120 ) ):
            trimmed,ox,oy,oz = trimBoard(frame[sl[0]])
            boardsDetected.append(trimmed)
            endMask[sl[0][0].start + ox:sl[0][0].start + ox + 8*oz,sl[0][1].start + oy:sl[0][1].start + oy + 8*oz] = 255      
    return endMask


def outputBoards(frame,mask,rng=[0,1]):
    cols = ['a','b','c','d','e','f','g','h']
    rows = ['8','7','6','5','4','3','2','1']
    labels, nb = ndimage.label(mask)
    boards = []
    for i in range(1,nb+1):
        sl = ndimage.find_objects(labels==i)
        board = frame[sl[0]]
        squareSizes = [int(x/8) for x in board.shape]
        pieces = dict()
        for x in range(8):
            for y in range(8):
                pieces[cols[x] + rows[y]] = np.array(PImage.fromarray(board[y*squareSizes[0]:(y+1)*squareSizes[0],x*squareSizes[1]:(x+1)*squareSizes[1]]).convert("L").resize([32,32], PImage.ADAPTIVE))
        boards.append(pieces)
    return boards


#########################################################################################################
## main programming section
#########################################################################################################

# check for correct usage
if len(sys.argv) < 2:
    print_usage()
link = sys.argv[1]
content_type = get_content_type(link)
if content_type != "Video":
	print_usage()

# download the video
video = downloadVideo(link)

# pick one frame in the middle to predict where the board is
pred_mask = predict_frame(video['frames'][int(len(video['frames'])/2)])
mask = reverseVideoMask(video['frames'][int(len(video['frames'])/2)],pred_mask)

# load the piece model used to predict pieces for each board
json_file = open('./pieceModel.json', 'r') # load json and create model
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('./pieceModel.h5') # load weights into new model

# loop over the frames and output the FEN string for each one
for i,frame in enumerate(video['frames']):
    
    # output the boards using the predicted mask as a template
    boards = outputBoards(frame,mask)

    # collect images for each square
    sq_names,sq_image = [],[]
    for sq,im in boards[0].items():
        sq_names.append(sq)
        sq_image.append(im.reshape(32,32,1))
    sq_image = np.array(sq_image)

    # collect predictions using the piece model
    predictions = dict()
    pieces = ['P','N','B','R','Q','K','p','n','b','r','q','k','E','E']
    for j in range(len(sq_image)):
        prediction = loaded_model.predict(sq_image[None,j],verbose=0)
        predictions[sq_names[j]] = pieces[np.argmax(prediction[0])]
        
    # set up a board using the predictions
    board = chess.Board(None)
    for square,piece in predictions.items():
        if piece != 'E':
            board.set_piece_at(chess.SQUARE_NAMES.index(square),chess.Piece.from_symbol(piece))

    # flip the board if necessary (could probably improve this code)
    bflip = chess.Board(None) 
    if ( len(chess.SquareSet(board.occupied_co[0])) > 0 ):
        if ( len(chess.SquareSet(board.occupied_co[1])) > 0 ):
            if ( sum(chess.SquareSet(board.occupied_co[0]))/len(chess.SquareSet(board.occupied_co[0])) < sum(chess.SquareSet(board.occupied_co[1]))/len(chess.SquareSet(board.occupied_co[1])) ):
                for square in chess.SquareSet(board.occupied_co[0]):
                    bflip.set_piece_at(chess.SQUARES[63-square],board.piece_at(square))
                for square in chess.SquareSet(board.occupied_co[1]):
                    bflip.set_piece_at(chess.SQUARES[63-square],board.piece_at(square))
                board = bflip

    # pring the result
    print("{}\t{}\t{}\t{}\t{}".format(video['title'],video['author'].replace(" ","_"),link+"&t="+str(i)+"s",chess.polyglot.zobrist_hash(board),board.fen()))

