from multiprocessing import Process
import sine_full
import plot_sine

if __name__ == '__main__':
    process1 = Process(target=sine_full.main)
    process2 = Process(target=plot_sine.animate)

    process1.start()
    process2.start()

    process1.join()
    process2.join()