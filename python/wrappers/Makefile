##################################################
# VLFeat Python Wrapping
##################################################

include ./MakefileBase

LIBRARY = libVLFeat-$(OS)-$(BITS).so
TARGET = ../vlfeat/_vlfeat.so

##################################################

VLFEAT_DIR = ../../vl

SOURCES_C = \
$(VLFEAT_DIR)/aib.c \
$(VLFEAT_DIR)/generic.c \
$(VLFEAT_DIR)/hikmeans.c \
$(VLFEAT_DIR)/ikmeans.c \
$(VLFEAT_DIR)/imopv.c \
$(VLFEAT_DIR)/mathop.c \
$(VLFEAT_DIR)/pgm.c \
$(VLFEAT_DIR)/rodrigues.c \
$(VLFEAT_DIR)/stringop.c \
$(VLFEAT_DIR)/getopt_long.c \
$(VLFEAT_DIR)/host.c \
$(VLFEAT_DIR)/imop.c \
$(VLFEAT_DIR)/imopv_sse2.c \
$(VLFEAT_DIR)/mser.c \
$(VLFEAT_DIR)/random.c \
$(VLFEAT_DIR)/sift.c 

SOURCES_CPP = \
vl_erfill_python.cpp \
vl_mser_python.cpp \
vl_sift_python.cpp \
pr_vlfeat.cpp

CFLAGS += -I$(VLFEAT_DIR)/../ -I../../../SubSystems -I/usr/include/python2.5
LFLAGS += -m32 -msse -shared 

OBJS_C=$(SOURCES_C:.c=.o)
OBJS_CPP=$(SOURCES_CPP:.cpp=.o)
OBJS =  $(OBJS_C) $(OBJS_CPP) 

.cpp.o:
	$(CXX) $(CFLAGS) -c $< -o $@

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@


.SUFFIXES:
.SUFFIXES:  .c .cpp .o

##################################################

.PHONY: all
all: $(LIBRARY)

.PHONY: install
install: $(LIBRARY) 
	cp $(LIBRARY) $(TARGET)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(LIBRARY)

##################################################

$(LIBRARY): $(OBJS)
	$(CXX) -o $@ $^ $(LFLAGS) -lboost_python-mt-py25

##################################################
