import React, {Component} from 'react';
import {recommend} from "../../libs/getJsonData";
import {Typography} from "antd";
import LoadingSpin from '../../common/js/loadingSpin';
import {ScrollMovieList} from "../../movieProfile/js/scrollMovieList";

const {Title} = Typography;

export class RecommendMovie extends Component {
    state = {
        movies: [],
        loading: false
    };
    fetchData = () => {
        const {id, type} = this.props;
        if (!id)
            return;
        this.setState({loading: true});
        const query = {id, type};
        recommend(query, movies => this.setState({movies, loading: false}));
    };

    componentDidMount() {
        this.fetchData();
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        const {id, type} = this.props;
        if (prevProps.id !== id || prevProps.type !== type)
            this.fetchData();
    }

    render() {
        const {movies, loading} = this.state;
        return (
            <div>
                <Title className='center margin-bottom' level={2}>根据影评，该用户可能对以下电影感兴趣</Title>
                {loading ? <LoadingSpin/> : <ScrollMovieList movies={movies}/>}
            </div>
        )
    }
}