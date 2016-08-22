TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
INCLUDEPATH += /usr/local/include/
LIBS += -L/usr/local/lib/ -lopencv_highgui -lopencv_core

SOURCES += main.cpp \
    nn.cpp

HEADERS += \
    nn.h \
    utils.h
