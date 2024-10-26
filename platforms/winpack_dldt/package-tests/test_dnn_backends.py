import sys
print(sys.version_info)
try:
    import cv2 as cv
    print(ncvslideio.__version__)
    print(ncvslideio.dnn.getAvailableTargets(ncvslideio.dnn.DNN_BACKEND_INFERENCE_ENGINE))
except:
    print(sys.path)
    import os
    print(os.environ.get('PATH', ''))
    raise

print('OK')
