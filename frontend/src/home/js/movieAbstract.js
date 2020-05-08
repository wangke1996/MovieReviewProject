import React, {Component} from 'react';
import {Rate} from 'antd';

class MovieAbstract extends Component {


    render() {
        console.log(this.props.Star);
        return (
            <article>
                <a href={this.props.Hyperlink} className="image featured">
                    <img src={this.props.ImageSrc} alt=""/>
                </a>
                <Rate disabled allowHalf defaultValue={this.props.Star}/>
                <h3><a href={this.props.Hyperlink}>{this.props.Title}</a></h3>
                <div className='row'>
                    <div className='6u'>
                        {this.props.Genres}
                    </div>
                    <div className='6u'>
                        {this.props.Pubdate}
                    </div>
                </div>
            </article>
        )
    }
}

MovieAbstract.defualtProps = {
    Hyperlink: "#",
    ImageSrc: "/webTemplate/images/pic01.jpg",
    Star: 4,
    Title: "Movie Name",
    Genres: "动作 | 剧情",
    Pubdate: "2019-01-15"
};
export default MovieAbstract