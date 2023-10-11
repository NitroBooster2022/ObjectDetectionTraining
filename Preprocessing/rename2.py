import os
import threading
def rename_file(file, num):
    image_dir = 'bg_new'
    old_image_name = os.path.join(image_dir, f'{file}.jpg')
    new_image_name = os.path.join(image_dir, f'{num}.jpg')
    os.rename(old_image_name, new_image_name)
def rename_files(start, end, files):
    image_dir = 'bg_new'
    label_dir = 'light_labels'

    for i in range(start, end):
        file_name, _ = os.path.splitext(files[i])

        old_image_name = os.path.join(image_dir, f'{file_name}.jpg')
        new_image_name = os.path.join(image_dir, f'{i}.jpg')
        os.rename(old_image_name, new_image_name)

#         old_label_name = os.path.join(label_dir, f'{file_name}.txt')
#         new_label_name = os.path.join(label_dir, f'{i}.txt')
#         os.rename(old_label_name, new_label_name)

def main():
    image_dir = 'bg_new'
    label_dir = 'light_labels'
#     files = [f for f in os.listdir(image_dir) if (f.endswith('.jpg') or f.endswith('.jpeg')or f.endswith('.JPG'))].
    files = [f for f in os.listdir(image_dir) if (f.endswith('.jpg'))]
    total_files = len(files)
    print("total: ", total_files)

    num_threads = 8
    files_per_thread = total_files // num_threads

    threads = []
    for i in range(num_threads):
        start = i * files_per_thread
        if i == num_threads - 1:
            end = total_files
        else:
            end = start + files_per_thread

        thread = threading.Thread(target=rename_files, args=(start, end, files))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
