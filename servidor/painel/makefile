# =========================
# CONFIGURAÇÃO
# =========================
CXX = g++
TARGET = ler_cartao
SRC = ler_cartao.cpp

INCLUDES = -I/usr/local/include
LIBS = -L/usr/local/lib \
       -lpteidlib -lpteidcommon -lpteidapplayer -lpteiddialogsQT \
       -lssl -lcrypto -lpthread -ldl

# =========================
# DEFAULT
# =========================
all: build

# =========================
# COMPILAR C++
# =========================
build:
	$(CXX) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS)

# =========================
# CORRER FLASK
# =========================
run:
	python3 app.py

# =========================
# BUILD + RUN
# =========================
start: build run

# =========================
# LIMPAR
# =========================
clean:
	rm -f $(TARGET)
