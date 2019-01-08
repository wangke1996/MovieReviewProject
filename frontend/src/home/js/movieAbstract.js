import React, {Component} from 'react';
import {Rate} from 'antd';

class MovieAbstract extends Component {
    constructor(props) {
        super(props);
    }



    render() {
        let prefix = process.env.PUBLIC_URL;
        return (
                <article>
                    <a href={this.props.Hyperlink}
                       className="image featured"><img
                        src={prefix + this.props.Sourcelink} alt=""/></a>
                    <Rate allowHalf defaultValue={this.props.Star}/>
                    <header>
                        <h3><a href={this.props.Hyperlink}>{this.props.Title}</a></h3>
                    </header>
                    <p>{this.props.Paragraph}</p>
                </article>
        )
    }
}

MovieAbstract.defualtProps = {
    Hyperlink: "#",
    Sourcelink: "/webTemplate/images/pic01.jpg",
    Star: 4,
    Titile: "Movie Name",
    Paragraph: "Description for this Movie"
};
export default MovieAbstract