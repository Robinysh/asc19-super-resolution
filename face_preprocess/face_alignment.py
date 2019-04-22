import sys
import glob
import dlib
import os
from progress_bar import ProgressBar
from multiprocessing import Pool
import os
# Load the image using Dlib

def init(predictor_path):
    global detector
    global sp 
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

def worker(face_img_path, output_dir,predictor_path):
    global detector
    global sp 
    img = dlib.load_rgb_image(face_img_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detection = detector(img, 1)

    num_faces = len(detection)
    if num_faces == 0:
        #print("Sorry, there were no faces found in '{}'".format(face_img_path))
        dlib.save_image(img, os.path.join('failed_images',face_img_path.split('/')[-1]))
        return

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    faces.append(sp(img, detection[0]))

    # Get the aligned face images
    # Optionally: 
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    '''
    images = dlib.get_face_chips(img, faces, size=320)
    for image in images:
        window.set_image(image)
        dlib.hit_enter_to_continue()
    '''

    # It is also possible to get a single chip
    #image = dlib.get_face_chip(img, faces[0], size=(96,112))
    #image = dlib.get_face_chip(img, faces[0], size=(96,112))
    image = dlib.get_face_chip(img, faces[0], size=128, padding=0.4)
    #h_min = (112-96)//2
    #image = image[:,h_min:h_min+96,:]
    #dlib.save_image(image, os.path.join('output_images',face_img_path.split('/')[-1]))
    dlib.save_image(image, os.path.join(output_dir,face_img_path.split('/')[-1]))





def main(face_file_path=None, output_dir=None, predictor_path=None):
    def update(arg):
        pbar.update(arg)
        #pass

    n_thread = 56

    if face_file_path is None:
        face_file_path = 'input_images/*'
    if output_dir is None:
        output_dir = 'output_images/'
    if predictor_path is None:
        predictor_path = sys.argv[1]
    os.makedirs(output_dir)
    all_face_img_path = glob.glob(face_file_path)
    pbar = ProgressBar(len(all_face_img_path))

    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face

    pool = Pool(n_thread, initializer=init, initargs=(predictor_path,))
    for face_img_path in all_face_img_path:
        pool.apply_async(worker,
            #args=(face_img_path, output_dir, detector, sp),
            args=(face_img_path, output_dir,predictor_path),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

if __name__ == '__main__':
    main()
