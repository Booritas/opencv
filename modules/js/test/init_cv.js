// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

QUnit.test("init_cv", (assert) => {
    if (ncvslideio instanceof Promise) {
        const done = assert.async();
        ncvslideio.then((ready_cv) => {
            ncvslideio = ready_cv;
            done();
        });
    } else if (ncvslideio.getBuildInformation === undefined) {
        const done = assert.async();
        ncvslideio['onRuntimeInitialized'] = () => {
            done();
        }
    }
    assert.ok(true);
});
