// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

QUnit.module('Core', {});

QUnit.test('test_LUT', function(assert) {
    // test LUT
    {
        let src = ncvslideio.matFromArray(3, 3, ncvslideio.CV_8UC1, [255, 128, 0, 0, 128, 255, 1, 2, 254]);
        let lutTable = [];
        for (let i = 0; i < 256; i++)
        {
           lutTable[i] = 255 - i;
        }
        let lut = ncvslideio.matFromArray(1, 256, ncvslideio.CV_8UC1, lutTable);
        let dst = new ncvslideio.Mat();

        ncvslideio.LUT(src, lut, dst);

        //console.log(dst.data);
        assert.equal(dst.ucharAt(0), 0);
        assert.equal(dst.ucharAt(1), 127);
        assert.equal(dst.ucharAt(2), 255);
        assert.equal(dst.ucharAt(3), 255);
        assert.equal(dst.ucharAt(4), 127);
        assert.equal(dst.ucharAt(5), 0);
        assert.equal(dst.ucharAt(6), 254);
        assert.equal(dst.ucharAt(7), 253);
        assert.equal(dst.ucharAt(8), 1);

        src.delete();
        lut.delete();
        dst.delete();
    }
});
