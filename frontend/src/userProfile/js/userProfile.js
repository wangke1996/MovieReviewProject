import React, {Component} from 'react';
import AvatarWithTags from './avatarWithTags'
import AbstractGraphs from './abstractGraphs'
import '../css/userProfile.css';

class UserProfile extends Component {
    render() {
        let prefix = process.env.PUBLIC_URL;
        return (

            <div id="Content" className="UserProfile">
                <div id="banner">
                    <h2>Hi! 欢迎使用 <strong>用户画像</strong>功能.</h2>
                    <span className="byline">
                        上传用户的影评，挖掘用户性格特点和电影审美
                    </span>
                    <hr/>
                </div>
                <div className="wrapper style1">
                    <section id="UserInfo">
                        <header>
                            <AvatarWithTags/>
                        </header>
                    </section>
                    <section id="AbstractGraphs">
                        <AbstractGraphs/>
                    </section>
                </div>
            </div>
        )
    }
}

export default UserProfile;
