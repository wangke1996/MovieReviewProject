const {override, fixBabelImports, addLessLoader} = require("customize-cra");

module.exports = override(
    fixBabelImports("import", {
        libraryName: "antd",
        libraryDirectory: "es",
        style: true
    }),
    addLessLoader({
        modifyVars: {"@primary-color": "#e89980"},
        javascriptEnabled: true,
    })
);