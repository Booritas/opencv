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

    const DilateSize = [cvSize.szQVGA, cvSize.szVGA, cvSize.szSVGA, cvSize.szXGA, cvSize.szSXGA];
    const DilateType = ["CV_8UC1", "CV_8UC4"];
    const combiDilate = combine(DilateSize, DilateType);

    function addDialteCase(suite, type) {
        suite.add('dilate', function() {
            ncvslideio.dilate(src, dst, kernel);
          }, {
              'setup': function() {
                let size = this.params.size;
                let matType = ncvslideio[this.params.matType];
                let src = new ncvslideio.Mat(size, matType);
                let dst = new ncvslideio.Mat(size, matType);
                let kernel = new ncvslideio.Mat();
                },
              'teardown': function() {
                src.delete();
                dst.delete();
                kernel.delete();
              }
          });
    }

    function addDilateModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let matType = combination[i][1];

        let params = {size: size, matType:matType};
        addKernelCase(suite, params, type, addDialteCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
      let suite = new Benchmark.Suite;
      totalCaseNum = 0;
      currentCaseId = 0;

      if (/\([0-9]+x[0-9]+,[\ ]*ncvslideio\_\w+\)/g.test(paramsContent.toString())) {
          let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*ncvslideio\_\w+\)/g)[0];
          let paramObjs = [];
          paramObjs.push({name:"size", value:"", reg:[""], index:0});
          paramObjs.push({name:"matType", value:"", reg:["/ncvslideio\_[0-9]+[FSUfsu]C[0-9]/"], index:1});
          let locationList = decodeParams2Case(params, paramObjs, dilateCombinations);

          for (let i = 0; i < locationList.length; i++){
              let first = locationList[i][0];
              let second = locationList[i][1];
              addDilateModeCase(suite, [dilateCombinations[first][second]], first);
            }
      } else {
        log("no filter or getting invalid params, run all the cases");
        addDilateModeCase(suite, combiDilate, 0);
      }
      setBenchmarkSuite(suite, "dilate", currentCaseId);
      log(`Running ${totalCaseNum} tests from dilate`);
      suite.run({ 'async': true }); // run the benchmark
  }

    let dilateCombinations = [combiDilate];

    if (isNodeJs) {
      const args = process.argv.slice(2);
      let paramsContent = '';
      if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*ncvslideio\_\w+\)/g.test(args.toString())) {
        paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*ncvslideio\_\w+\)/g)[0];
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