/**
 * Created by 王颗 on 2019/2/27.
 */
import reqwest from 'reqwest';

function fetchData(url, callback) {
    reqwest({
        url: url,
        type: 'json',
        method: 'get',
        contentType: 'application/json',
        success: (res) => {
            callback(res);
        },
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
    let url = '/getMovieReviews/' + movieID + '/' + count;
    fetchData(url, callback);
}

export function getMovieComments(movieID, callback, count = 100) {
    let url = '/getMovieComments/' + movieID + '/' + count;
    fetchData(url, callback);
}

export function getMoviePhotos(movieID, callback, count = 100) {
    let url = '/getMoviePhotos/' + movieID + '/' + count;
    fetchData(url, callback);
}

export function getMovieReviewsTrend(movieID, callback) {
    let url = '/getMovieReviewsTrend/' + movieID;
    fetchData(url, callback);
}
