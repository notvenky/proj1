import threading
import sine_full
import plot_sine

def run_concurrently():
    thread1 = threading.Thread(target=sine_full.main)
    thread2 = threading.Thread(target=plot_sine.animate)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

if __name__ == "__main__":
    run_concurrently()