/**
 * Created by 王颗 on 2019/2/27.
 */
import reqwest from 'reqwest';

const backendURL = 'http://101.6.69.26:5002';

function fetchData(url, callback, data = {}) {
    reqwest({
        url: backendURL + url,
        type: 'json',
        method: 'get',
        contentType: 'application/json',
        data: data,
        success: callback,
    });
}

export function getMovieInTheater(callback) {
    let url = '/getMovieInTheater';
    fetchData(url, callback);
}

export function getMovieInfo(movieID, callback) {
    let url = '/getMovieInfo/' + movieID;
    fetchData(url, callback);
}

export function getMovieReviews(movieID, callback, count = 100) {
    const url = '/getMovieReviews/' + movieID + '/' + count;
    fetchData(url, callback);
}

export function getMovieComments(movieID, callback, count = 100) {
    const url = '/getMovieComments/' + movieID + '/' + count;
    fetchData(url, callback);
}

export function getMoviePhotos(movieID, callback, count = 100) {
    const url = '/getMoviePhotos/' + movieID + '/' + count;
    fetchData(url, callback);
}

export function getMovieReviewsTrend(movieID, callback) {
    const url = '/getMovieReviewsTrend/' + movieID;
    fetchData(url, callback);
}

export function getTargetFreqs(movieID, target, callback) {
    const url = '/getTargetFreqs/' + movieID + '/' + target;
    console.log(url);
    fetchData(url, callback);
}

export function getRelatedSentences(query, callback) {
    const url = '/getRelatedSentences';
    fetchData(url, callback, query)
}

export function getTargetList(query, callback) {
    const url = '/getTargetList';
    fetchData(url, callback, query);
}

export function getTargetDetail(query, callback) {
    const url = '/getTargetDetail';
    fetchData(url, callback, query);
}

export function searchTarget(movieID, input_value, callback) {
    if (!input_value) {
        callback([]);
        return;
    }
    const url = '/searchTarget/' + movieID + '/' + input_value;
    fetchData(url, callback);
}