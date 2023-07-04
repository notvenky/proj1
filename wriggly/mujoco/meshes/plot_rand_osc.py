import threading
import rand_sine
import plot_sine

def run_concurrently():
    thread1 = threading.Thread(target=rand_sine.main)
    thread2 = threading.Thread(target=plot_sine.animate)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

if __name__ == "__main__":
    run_concurrently()