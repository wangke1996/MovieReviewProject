import React, {Component} from 'react'
import MovieAbstract from './movieAbstract'
import {getMovieInTheater} from '../../libs/getJsonData'
import {image_url} from '../../libs/toolFunctions'
import LoadingSpin from "../../common/js/loadingSpin";
import '../css/movieInTheater.css'
import HorizontalScroll from 'react-scroll-horizontal'
import {ScrollMovieList} from "../../movieProfile/js/scrollMovieList";

class MovieInTheater extends Component {
    state = {
        moviesData: [],
        loadedFlag: false,
    };

    loadData() {
        getMovieInTheater((data) => {
            this.setState({
                moviesData: data,
                loadedFlag: true
            })
        });
    }

    componentDidMount() {
        this.loadData();
    }

    render() {
        if (!this.state.loadedFlag)
            return (<LoadingSpin tip='数据在线爬取中，可能需要数分钟时间'/>);
        // const moviesElement = [];
        // this.state.moviesData.forEach((d, i) => moviesElement.push(
        //     <MovieAbstract key={i} Hyperlink={'/movieProfile/' + d.id} ImageSrc={image_url(d.images.medium)}
        //                    Star={Math.round(10 * d.rating.average / d.rating.max) / 2} Title={d.title}
        //                    Genres={d.genres.join(' | ')} Pubdate={d.pubdates[d.pubdates.length - 1].slice(0, 10)}/>
        // ));
        console.log(this.state);
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
                {/*<div className='scrollReel'>*/}
                {/*    <HorizontalScroll reverseScroll={true}>*/}
                {/*        {moviesElement}*/}
                {/*    </HorizontalScroll>*/}
                {/*</div>*/}
                <ScrollMovieList movies={this.state.moviesData.map(d => d.id)}/>
                <hr/>
            </div>

        )
    }
}

export default MovieInTheater;