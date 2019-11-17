package com.amitshekhar.tflite;

import android.graphics.Bitmap;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    Recognition recognizeImage(Bitmap bitmap);
    Bitmap convertBitmapIntoGreyscale(Bitmap bitmap);

    void close();

    class Recognition {


        private final int digit1;
        private final int digit2;
        private final int digit3;
        private final int length;


        public Recognition(
                final int digit1, final int digit2, final int digit3, final int length) {
            this.digit1 = digit1;
            this.digit2 = digit2;
            this.digit3 = digit3;
            this.length = length;

        }

        public int getLength() {
            return length;
        }

        @Override
        public String toString() {
            String resultString = "";
            String is = "";

            if (length == 0)
                return "There is no number in the picture!";
            if (digit1 != 10) {
                resultString += Integer.toString(digit1);
                is += "1";
            }
            if (digit2 != 10) {
                resultString += Integer.toString(digit2);
                is += "2";
            }
            if (digit3 != 10) {
                resultString += Integer.toString(digit3);
                is += "3";
            }

            //resultString += " (" + is + ")"

            return resultString.trim();
        }
    }
}
