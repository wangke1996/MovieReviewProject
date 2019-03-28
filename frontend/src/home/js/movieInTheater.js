import React, {Component} from 'react'
import MovieAbstract from './movieAbstract'
import {getMovieInTheater} from '../../libs/getJsonData'
import {image_url} from '../../libs/toolFunctions'
import LoadingSpin from "../../common/js/loadingSpin";
import '../css/movieInTheater.css'
import HorizontalScroll from 'react-scroll-horizontal'

class MovieInTheater extends Component {

    loadData() {
        getMovieInTheater((data) => {
            this.setState({
                moviesData: data,
                loadedFlag: true
            })
        });
    }

    constructor(props) {
        super(props);
        this.state = {
            moviesData: [],
            loadedFlag: false,
        };
        this.loadData();
    }


    render() {
        if (!this.state.loadedFlag)
            return (<LoadingSpin tip='数据在线爬取中，可能需要数分钟时间'/>);
        const moviesElement = [];
        this.state.moviesData.forEach(d => moviesElement.push(
            <MovieAbstract Hyperlink={'/movieProfile/' + d.id} ImageSrc={image_url(d.images.medium)}
                           Star={Math.round(10 * d.rating.average / d.rating.max) / 2} Title={d.title}
                           Genres={d.genres.join(' | ')} Pubdate={d.pubdates[d.pubdates.length-1].slice(0,10)}/>
        ));
        return (

            <div className="LatestMovie wrapper style2">
                <article className="container special">
                    <header>
                        <h2>最新电影</h2>
                        <span className="byline">
                            点击图片查看<strong>最新影评</strong>及该电影<strong>整体风评</strong>
                        </span>
                    </header>
                </article>
                <div className='scrollReel'>
                    <HorizontalScroll reverseScroll={true}>
                        {moviesElement}
                    </HorizontalScroll>
                </div>
                <hr/>
            </div>

        )
    }
}

export default MovieInTheater;