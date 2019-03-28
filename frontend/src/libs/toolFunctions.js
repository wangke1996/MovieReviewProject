export function image_url(url) {
    return 'https://images.weserv.nl/?url=' + url;
}

export function isEmpty(obj) {
    return Object.keys(obj).length === 0;
}