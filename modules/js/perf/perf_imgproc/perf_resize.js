var isNodeJs = (typeof window) === 'undefined'? true : false;

if　(isNodeJs)　{
  var Benchmark = require('benchmark');
  var ncvslideio = require('../../opencv');
  var HelpFunc = require('../perf_helpfunc');
  var Base = require('../base');
} else {
  var paramsElement = document.getElementById('params');
  var runButton = document.getElementById('runButton');
  var logElement = document.getElementById('log');
}

function perf() {

  console.log('opencv.js loaded');
  if (isNodeJs) {
    global.ncvslideio = ncvslideio;
    global.combine = HelpFunc.combine;
    global.fillGradient = HelpFunc.fillGradient;
    global.log = HelpFunc.log;
    global.decodeParams2Case = HelpFunc.decodeParams2Case;
    global.setBenchmarkSuite = HelpFunc.setBenchmarkSuite;
    global.addKernelCase = HelpFunc.addKernelCase;
    global.cvSize = Base.getCvSize();
  } else {
    enableButton();
    cvSize = getCvSize();
  }
  let totalCaseNum, currentCaseId;

  const matTypesUpLinear = ['CV_8UC1', 'CV_8UC2', 'CV_8UC3', 'CV_8UC4'];
  const size1UpLinear = [cvSize.szVGA];
  const size2UpLinear = [cvSize.szqHD, cvSize.sz720p];
  const combiUpLinear = combine(matTypesUpLinear, size1UpLinear, size2UpLinear);

  const combiDownLinear = [
    ['CV_8UC1', cvSize.szVGA, cvSize.szQVGA],
    ['CV_8UC2', cvSize.szVGA, cvSize.szQVGA],
    ['CV_8UC3', cvSize.szVGA, cvSize.szQVGA],
    ['CV_8UC4', cvSize.szVGA, cvSize.szQVGA],
    ['CV_8UC1', cvSize.szqHD, cvSize.szVGA],
    ['CV_8UC2', cvSize.szqHD, cvSize.szVGA],
    ['CV_8UC3', cvSize.szqHD, cvSize.szVGA],
    ['CV_8UC4', cvSize.szqHD, cvSize.szVGA],
    ['CV_8UC1', cvSize.sz720p, cvSize.sz213x120],// face detection min_face_size = 20%
    ['CV_8UC2', cvSize.sz720p, cvSize.sz213x120],// face detection min_face_size = 20%
    ['CV_8UC3', cvSize.sz720p, cvSize.sz213x120],// face detection min_face_size = 20%
    ['CV_8UC4', cvSize.sz720p, cvSize.sz213x120],// face detection min_face_size = 20%
    ['CV_8UC1', cvSize.sz720p, cvSize.szVGA],
    ['CV_8UC2', cvSize.sz720p, cvSize.szVGA],
    ['CV_8UC3', cvSize.sz720p, cvSize.szVGA],
    ['CV_8UC4', cvSize.sz720p, cvSize.szVGA],
    ['CV_8UC1', cvSize.sz720p, cvSize.szQVGA],
    ['CV_8UC2', cvSize.sz720p, cvSize.szQVGA],
    ['CV_8UC3', cvSize.sz720p, cvSize.szQVGA],
    ['CV_8UC4', cvSize.sz720p, cvSize.szQVGA]
  ];

  const matTypesAreaFast = ['CV_8UC1', 'CV_8UC3', 'CV_8UC4', 'CV_16UC1', 'CV_16UC3', 'CV_16UC4'];
  const sizesAreaFast = [cvSize.szVGA, cvSize.szqHD, cvSize.sz720p, cvSize.sz1080p];
  const scalesAreaFast = [2];
  const combiAreaFast = combine(matTypesAreaFast, sizesAreaFast, scalesAreaFast);

  function addResizeCase(suite, type) {
    suite.add('resize', function() {
      if (type == "area") {
        ncvslideio.resize(src, dst, dst.size(), 0, 0, ncvslideio.INTER_AREA);
      } else {
        ncvslideio.resize(src, dst, to, 0, 0, ncvslideio.INTER_LINEAR_EXACT);
      }
    }, {
        'setup': function() {
          let from = this.params.from;
          let to = this.params.to;
          let matType = ncvslideio[this.params.matType];
          let src = new ncvslideio.Mat(from, matType);
          let type = this.params.modeType;
          let dst;
          if (type == "area") {
            dst = new ncvslideio.Mat(from.height/scale, from.width/scale, matType);
          } else {
            dst = new ncvslideio.Mat(to, matType);
            fillGradient(ncvslideio, src);
          }
          },
        'teardown': function() {
          src.delete();
          dst.delete();
        }
    });
  }

  function addResizeModeCase(suite, combination, type) {
    totalCaseNum += combination.length;
    for (let i = 0; i < combination.length; ++i) {
      let matType = combination[i][0];
      let from = combination[i][1];
      let params;
      if (type == "area") {
        let scale = combination[i][2];
        params = { from: from, scale: scale, matType: matType, modeType: type };
      } else {
        let to = combination[i][2];
        params = { from: from, to: to, matType: matType, modeType: type};
      }
      addKernelCase(suite, params, type, addResizeCase)
    }
  }

  function genBenchmarkCase(paramsContent) {
    let suite = new Benchmark.Suite;
    totalCaseNum = 0;
    currentCaseId = 0;
    if (/\(\w+,[\ ]*[0-9]+x[0-9]+,[\ ]*[0-9]+x[0-9]+\)/g.test(paramsContent.toString())) {
      let params = paramsContent.toString().match(/\(\w+,[\ ]*[0-9]+x[0-9]+,[\ ]*[0-9]+x[0-9]+\)/g)[0];
      let paramObjs = [];
      paramObjs.push({name:"matType", value:"", reg:["/ncvslideio\_[0-9]+[A-z][A-z][0-9]/"], index:0});
      paramObjs.push({name:"size1", value:"", reg:[""], index:1});
      paramObjs.push({name:"size2", value:"", reg:[""], index:2});
      let locationList = decodeParams2Case(params, paramObjs,combinations);

      for (let i = 0; i < locationList.length; i++){
        let first = locationList[i][0];
        let second = locationList[i][1];
        addResizeModeCase(suite, [combinations[first][second]], "linear");
      }
    } else {
      log("no filter or getting invalid params, run all the cases");
      addResizeModeCase(suite, combiUpLinear, "linear");
      addResizeModeCase(suite, combiDownLinear, "linear");
    }
    setBenchmarkSuite(suite, "resize", currentCaseId);
    log(`Running ${totalCaseNum} tests from Resize`);
    suite.run({ 'async': true }); // run the benchmark
  }

  // init
  let combinations = [combiUpLinear, combiDownLinear];//, combiAreaFast];

  // set test filter params
  if (isNodeJs) {
    const args = process.argv.slice(2);
    let paramsContent = '';
    if (/--test_param_filter=\(\w+,[\ ]*[0-9]+x[0-9]+,[\ ]*[0-9]+x[0-9]+\)/g.test(args.toString())) {
      paramsContent = args.toString().match(/\(\w+,[\ ]*[0-9]+x[0-9]+,[\ ]*[0-9]+x[0-9]+\)/g)[0];
    }
    genBenchmarkCase(paramsContent);
  } else {
    runButton.onclick = function()　{
      let paramsContent = paramsElement.value;
      genBenchmarkCase(paramsContent);
      if (totalCaseNum !== 0) {
        disableButton();
      }
    }
  }
};

async function main() {
  if (ncvslideio instanceof Promise) {
    ncvslideio = await ncvslideio;
    perf();
  } else {
    ncvslideio.onRuntimeInitialized = perf;
  }
}

main();