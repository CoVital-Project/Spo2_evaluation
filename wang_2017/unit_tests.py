import wang_2017
import numpy as np
import unittest

class TestRemovePeak(unittest.TestCase):

    def test_under(self):
        print("\n\nTest under")
        wang = wang_2017.Wang2017()
        
        diff_between_peaks = np.array([1, 1, 1])
        peaks = np.array([10, 11, 12, 13])
        
        diff_between_peaks_original = diff_between_peaks.copy()
        peaks_original = peaks.copy()
        
        diff_between_peaks_tmp, peaks_tmp, _ = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, 0, 0.5)
        np.testing.assert_array_equal(diff_between_peaks_tmp, diff_between_peaks)
        
        print("Second test")
        diff_between_peaks_tmp, peaks_tmp, _ = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, 1, 2)
        np.testing.assert_array_equal(np.array([1, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(diff_between_peaks_original, diff_between_peaks)
        np.testing.assert_array_equal(np.array([11, 12, 13]), peaks_tmp)
        
        print("Third test")
        diff_between_peaks_tmp, peaks_tmp, _ = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, 0, 2)
        np.testing.assert_array_equal(np.array([1, 1, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(np.array([10, 11, 12, 13]), peaks_tmp)
        
        print("Third test")
        diff_between_peaks_tmp, peaks_tmp, _ = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, 3, 3)
        np.testing.assert_array_equal(np.array([3]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(np.array([10, 13]), peaks_tmp)
        
        diff_between_peaks = np.array([0.3, 0.7, 1])
        peaks = np.array([10, 10.3, 11, 12])
        diff_between_peaks_original = diff_between_peaks.copy()
        peaks_original = peaks.copy()
        
        diff_between_peaks_tmp, peaks_tmp, _ = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, 1, 1)
        np.testing.assert_array_equal(np.array([0.7, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(diff_between_peaks_original, diff_between_peaks)
        np.testing.assert_array_equal(np.array([10.3, 11, 12]), peaks_tmp)
        
        diff_between_peaks_tmp, peaks_tmp, _ = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, 2, 1)
        np.testing.assert_array_equal(np.array([1, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(diff_between_peaks_original, diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 12]), peaks_tmp)
        
    def test_above(self):
        print("\n\nTest above")
        
        wang = wang_2017.Wang2017()
        
        diff_between_peaks = np.array([1, 1, 1])
        peaks = np.array([10, 11, 12, 13])
        
        diff_between_peaks_original = diff_between_peaks.copy()
        peaks_original = peaks.copy()
        
        diff_between_peaks_tmp, peaks_tmp = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, 0, 0.5)
        np.testing.assert_array_equal(diff_between_peaks_tmp, diff_between_peaks)
        
        print("Second test")
        diff_between_peaks_tmp, peaks_tmp = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, 1, 2)
        np.testing.assert_array_equal(np.array([1, 2]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(diff_between_peaks_original, diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 13]), peaks_tmp)
        
        print("Third test")
        diff_between_peaks_tmp, peaks_tmp = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, 0, 2)
        np.testing.assert_array_equal(np.array([2, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(np.array([10, 12, 13]), peaks_tmp)
        
        print("Third test")
        diff_between_peaks_tmp, peaks_tmp = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, 3, 3)
        np.testing.assert_array_equal(np.array([1, 1, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(np.array([10, 11, 12, 13]), peaks_tmp)  
        
        diff_between_peaks = np.array([1, 0.7, 0.3])
        peaks = np.array([10, 11, 11.7, 12])
        diff_between_peaks_original = diff_between_peaks.copy()
        peaks_original = peaks.copy()
        
        diff_between_peaks_tmp, peaks_tmp = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, 1, 1)
        np.testing.assert_array_equal(np.array([1, 1]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(diff_between_peaks_original, diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 12]), peaks_tmp)
        
        diff_between_peaks_tmp, peaks_tmp = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, 2, 1)
        np.testing.assert_array_equal(np.array([1, 0.7]), diff_between_peaks_tmp)
        np.testing.assert_array_equal(diff_between_peaks_original, diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 11.7]), peaks_tmp)
    
    
    
    
    def test_remove_all_peaks(self):
        print("\n\nTest All peaks")
        wang = wang_2017.Wang2017()
        
        diff_between_peaks = np.array([1, 0.2, 0.8, 0.8, 0.2, 0.5, 0.5])
        peaks = np.array([10, 11, 11.2, 12, 12.8, 13, 13.5, 14])
        
        main_peak_position = 3
        heart_rate_in_sec = 1
        while main_peak_position < len(peaks):
            diff_between_peaks, peaks = wang.remove_extraneous_peaks_above_a_peak(diff_between_peaks, peaks, main_peak_position, heart_rate_in_sec)
            main_peak_position += 1
        
        np.testing.assert_array_equal(np.array([1, 0.2, 0.8, 1, 1]), diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 11.2, 12, 13, 14]), peaks)
        
        main_peak_position = 3
        while main_peak_position > 0:
            print("Psotion ", main_peak_position)
            diff_between_peaks, peaks, main_peak_position = wang.remove_extraneous_peaks_under_a_peak(diff_between_peaks, peaks, main_peak_position, heart_rate_in_sec)
            main_peak_position -= 1
        
        np.testing.assert_array_equal(np.array([1, 1, 1, 1]), diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 12, 13, 14]), peaks)
        
        print("\n\nMAIN")
        
        diff_between_peaks = np.array([1, 0.2, 0.8, 0.8, 0.2, 0.5, 0.5])
        peaks = np.array([10, 11, 11.2, 12, 12.8, 13, 13.5, 14])
        main_peak_position = 3
        diff_between_peaks, peaks, main_peak_position = wang.remove_extraneous_peaks_from_a_peak(diff_between_peaks, peaks, main_peak_position, heart_rate_in_sec)
        
        np.testing.assert_array_equal(np.array([1, 1, 1, 1]), diff_between_peaks)
        np.testing.assert_array_equal(np.array([10, 11, 12, 13, 14]), peaks)
        self.assertEqual(main_peak_position, 2)
        
        
        
    def test_remove_from_ppg(self):
        print("\n\nRemove from PPG")
        wang = wang_2017.Wang2017()
        
        
        ppg = np.array([0 , 0, 0.2, 0.1, 1, 0, 0, 0.4, 1.5, 0, 0.5, 0.3, 1])
        peaks = np.array([2, 4, 8, 10, 12])
        timestamps = np.array([200, 400, 800, 1000, 1200])
        heart_rate_in_millisec = 400
        peaks_ret = wang.remove_extraneous_peaks(ppg, peaks, timestamps, heart_rate_in_millisec)
        np.testing.assert_array_equal(np.array([4, 8, 12]), peaks_ret)

    #def test_isupper(self):
        #self.assertTrue('FOO'.isupper())
        #self.assertFalse('Foo'.isupper())

    #def test_split(self):
        #s = 'hello world'
        #self.assertEqual(s.split(), ['hello', 'world'])
        ## check that s.split fails when the separator is not a string
        #with self.assertRaises(TypeError):
            #s.split(2)

if __name__ == '__main__':
    unittest.main()

