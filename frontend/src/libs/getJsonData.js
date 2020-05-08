/**
 * Created by 王颗 on 2019/2/27.
 */
import reqwest from 'reqwest';
import {backendURL, debug_mode} from "./config";

export function wrapUrl(url, randParam = false) {
    let trueURL;
    if (debug_mode)
        trueURL = backendURL + url;
    else
        trueURL = url;
    if (randParam)
        trueURL += '?time=' + (new Date().getTime());
    return trueURL;
}

function fetchData(url, callback, data = {}) {
    reqwest({
        url: wrapUrl(url),
        type: 'json',
        method: 'get',
        contentType: 'application/json',
        data: data,
        success: callback,
        error: callback
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

export function getTargetFreqs(query, callback) {
    const url = '/getTargetFreqs';
    fetchData(url, callback, query);
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

export function searchTarget(query, callback) {
    if (!query.input) {
        callback([]);
        return;
    }
    const url = '/searchTarget';
    fetchData(url, callback, query);
}

export function getUserProfile(uid, callback) {
    const url = '/getUserProfile/' + uid;
    fetchData(url, callback);
}

export function searchUser(query, callback) {
    const url = '/searchUser/' + query;
    fetchData(url, callback)
}

export function getActiveUsers(callback) {
    const url = '/getActiveUsers';
    fetchData(url, callback);
}

export function checkUserState(uid, callback) {
    const url = '/checkUserState/' + uid;
    fetchData(url, callback);
}

export function analysisUploadedFile(file, callback) {
    fetchData('/analysisUploadedFile', callback, {file: file});
}

export function analysisReview(text, callback) {
    fetchData('/analysisReview', callback, {text: text})
}

export function getDemo(callback) {
    fetchData('/getDemo', (res) => callback(res.response));
}

export function recommend(query, callback) {
    fetchData('/recommend', callback, query)
}

export function download(cacheId, callback) {
    fetchData('/download/' + cacheId, callback);
}