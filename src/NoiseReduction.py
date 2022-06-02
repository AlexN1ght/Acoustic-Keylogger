from scipy.io import wavfile
import noisereduce as nr

file_name = input()

rate, data = wavfile.read(file_name)

# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write(f"{file_name.split('.')[0]}_cleared.{file_name.split('.')[1]}", rate, reduced_noise)

