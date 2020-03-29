import React, {Component} from 'react';
import {Card, Tooltip, Typography} from 'antd';
import {image_url} from "../../libs/toolFunctions";

const {Paragraph} = Typography;
const {Meta} = Card;

class FavoriteActorInfo extends Component {
    render() {
        const {flag, name, description, img, saw, url, text} = this.props;
        return (
            <div className={"6u " + flag} id="favoriteActor">
                <header>
                    <h2>最喜欢的演员是<span className="emphatic">{name}</span></h2>
                    <span className="byline">看了{saw}部Ta主演的电影 {text}</span>
                </header>
                <Tooltip title='点击查看该演员详细信息'>
                    <a href={url}>
                        <Card
                            hoverable
                            style={{width: 240}}
                            cover={<img alt={name} src={image_url(img)}/>}
                        >
                            <Meta
                                title={name}
                                // description={description}
                            />
                        </Card>
                        <Paragraph className='9u center' style={{float: 'none', textAlign: 'left', textIndent:'2em'}}>{description}</Paragraph>
                    </a>
                </Tooltip>
            </div>
        )
    }
}

export default FavoriteActorInfo