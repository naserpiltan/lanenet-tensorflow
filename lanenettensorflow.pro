QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/home/eagle-soft/opencv/build/lib\
     -lopencv_core\
     -lopencv_highgui \
     -lopencv_imgcodecs \
     -lopencv_imgproc \
     -lopencv_flann \
     -lopencv_photo \
     -lopencv_videoio \
     -lopencv_video \
     -lopencv_tracking \
     -lopencv_dnn \
     -lopencv_xfeatures2d \
     -lopencv_features2d \
     -lopencv_calib3d




INCLUDEPATH += /opt/intel/openvino/deployment_tools/inference_engine/include
INCLUDEPATH +=/opt/intel/openvino/deployment_tools/inference_engine/samples/cpp
LIBS +=-L/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64 \
    -linference_engine \
    -linference_engine_nn_builder \
    -linference_engine_legacy \
    -linference_engine_transformations

LIBS +=-L/opt/intel/openvino_2020.2.120/deployment_tools/ngraph/lib \
    -lngraph
LIBS +=-L/opt/intel/openvino_2020.2.120/deployment_tools/inference_engine/external/tbb/lib \
    -ltbb
CUDA_DIR = /usr/local/cuda
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart -lcuda

HEADERS += \
    monodepth.h \
    my_utils.h

INCLUDEPATH += /home/eagle-soft/tensorflow/local/include/google/tensorflow
INCLUDEPATH += /home/eagle-soft/tensorflow/bazel-tensorflow/external/nsync/public
INCLUDEPATH += /usr/include/eigen3
INCLUDEPATH += /home/eagle-soft/tensorflow/tensorflow/contrib/makefile/downloads/absl


LIBS +=-L/home/eagle-soft/tensorflow/local/lib \
     -ltensorflow_all
