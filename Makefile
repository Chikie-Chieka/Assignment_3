CC = gcc
CFLAGS = -O3 -Wall -Wextra -Iinclude -I/usr/local/include
LDFLAGS = -L/usr/local/lib -loqs -lcrypto -lpthread -lm

# Source files
SRCS = src/main.c \
       src/util.c \
       src/csv.c \
       src/models.c \
       src/model_oqs_kem.c \
       src/model_ascon80pq.c \
       src/ent_phase.c \

# Object files
OBJS = $(SRCS:.c=.o)

# Target binary
TARGET = bench_c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)