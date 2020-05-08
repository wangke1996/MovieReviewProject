import React, {Component} from 'react';
import {getMovieInfo} from "../../libs/getJsonData";
import {image_url} from "../../libs/toolFunctions";
import MovieAbstract from "../../home/js/movieAbstract";
import HorizontalScroll from "react-scroll-horizontal";

class MovieCard extends Component {
    state = {
        data: null,
    };

    fetchData = () => {
        const {id} = this.props;
        if (!id)
            return;
        getMovieInfo(id, data => this.setState({data}));
    };

    componentDidMount() {
        this.fetchData();
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevProps.id !== this.props.id)
            this.fetchData();
    }

    render() {
        const {data} = this.state;
        if (!data || !data.id)
            return [];
        console.log(data);
        const {id, images, rating, title, genres, pubdates} = data;
        return (
            <MovieAbstract Hyperlink={'/movieProfile/' + id} ImageSrc={image_url(images.medium)}
                           Star={Math.round(10 * rating.average / rating.max) / 2} Title={title}
                           Genres={genres.join(' | ')} Pubdate={(pubdates[0]||'').slice(0, 10)}/>

        )
    }
}

export class ScrollMovieList extends Component {
    render() {
        const {movies} = this.props;
        console.log(movies);
        if (!movies || !movies.length)
            return [];
        return (
            <div className='scrollReel'>
                <HorizontalScroll reverseScroll={true}>
                    {movies.map((id, i) => <MovieCard key={i} id={id}/>)}
                </HorizontalScroll>
            </div>
        )
    }
}