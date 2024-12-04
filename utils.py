import numpy as np
import struct

def read_database(data_size, vec_size):
    data = []
    with open('saved_db.dat', 'rb') as file:
        # Calculate the position of the row where the size of one row (vec_size * float_size)
        for row_id in range(data_size):
            position = row_id * (vec_size * 4)
            # Seek to the position of the row
            file.seek(position)
            # Read the row with size of one row (ID + vec_size * floats)
            packed_data = file.read(vec_size * 4)
            data.append(struct.unpack(f'{vec_size}f', packed_data))
    return data
    