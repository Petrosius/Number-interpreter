package com.amitshekhar.tflite;

import android.graphics.Bitmap;


public interface RegressionModel {

    Recognition recognizeImage(Bitmap bitmap);

    void close();

    class Recognition {

        private float top;
        private float left;
        private float width;
        private float height;
        private int bitmapHeight;
        private int bitmapWidth;


        public Recognition(
                final float top, final float left,
                final float height, final float width) {
            this.top = top;
            this.left = left;
            this.height = height;
            this.width = width;
        }

        public void setBitmapSize(int height, int width){
            bitmapHeight = height;
            bitmapWidth = width;
        }



        public int getBitmapHeight() {
            return bitmapHeight;
        }

        public int getBitmapWidth() {
            return bitmapWidth;
        }

        public int getTop() {
            return (int) (top * bitmapHeight);
        }

        public int getBottom() {
            return (int) ((top + height) * bitmapHeight);
        }

        public int getRight() {
            return (int) ((left + width) * bitmapWidth) ;
        }

        public int getLeft() {
            return (int) (left* bitmapWidth);
        }

        public int getHeight() {
            return (int) (height * bitmapHeight);
        }

        public int getWidth(){
            return (int) (width * bitmapWidth);
        }

        private float getUnnormalizedWidthPixel(float number) {
            return number * 64;
        }

        private float getUnnormalizedHeightPixel(float number) {
            return number * 48;
        }

        public String toString(boolean printCoordinates) {
            String resultString = "";
            if(printCoordinates)
                resultString =  String.format("\n (Top,Bottom,Right,Left) = (%d,%d,%d,%d)",
                    getTop(), getBottom(), getRight(), getLeft());
            return resultString.trim();
        }
    }
}
