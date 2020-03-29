export function image_url(url) {
    return 'https://images.weserv.nl/?url=' + url;
}

export function isEmpty(obj) {
    return Object.keys(obj).length === 0;
}

export function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}

export function toPercent(val, decimals = 0) {
    return (val * 100).toFixed(decimals) + '%';
}