CC = gcc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS = -lm -lpthread

TARGET = galton_board
SRCS = main.c
HEADERS = stb_image_write.h

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) galton_board.png

run: $(TARGET)
	./$(TARGET) --balls 100000

.PHONY: clean run
