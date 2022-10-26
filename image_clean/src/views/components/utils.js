const getImageObjKeys = (imgObj) => {
    return imgObj.value.filter(e => e.select).map(e => e.key)
}
const getNameFromPath = (path) => {
    return path.substr(path.lastIndexOf('/') + 1)
}
export { getImageObjKeys, getNameFromPath}