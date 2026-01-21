.PHONY: clean

CXX = g++
# should i put some compiler flags in?
CXXFLAGS = -pthread -O3 -Iinclude

SRCDIR = src
OBJDIR = build
BINDIR = bin

TARGET = $(BINDIR)/main

OBJS = \
	$(OBJDIR)/main.o \
	$(OBJDIR)/CTRNN.o \
	$(OBJDIR)/TSearch.o \
	$(OBJDIR)/CountingAgent.o \
	$(OBJDIR)/random.o

$(TARGET): $(OBJS)
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp include/CTRNN.h include/CountingAgent.h include/TSearch.h
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/CTRNN.o: $(SRCDIR)/CTRNN.cpp include/CTRNN.h include/random.h
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/TSearch.o: $(SRCDIR)/TSearch.cpp include/TSearch.h
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/CountingAgent.o: $(SRCDIR)/CountingAgent.cpp include/CountingAgent.h include/TSearch.h include/CTRNN.h include/random.h include/VectorMatrix.h
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/random.o: $(SRCDIR)/random.cpp include/random.h include/VectorMatrix.h
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o main
