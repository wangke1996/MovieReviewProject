import React, {Component} from 'react';
import {Button} from 'antd';
import '../css/userProfile.css';

class UserProfile extends Component {
    render() {
        let prefix = process.env.PUBLIC_URL;
        return (
            <div id="Content" className="Home">
                <Button type='primary'>Button</Button>
                <div className="wrapper style2">
                    <article className="container special">
                        <header>
                            <h2>最新电影</h2>
                            <span className="byline">
                                点击图片查看<strong>最新影评</strong>及该电影<strong>整体风评</strong>
                            </span>
                        </header>
                    </article>
                    <div className="carousel">
                        <div className="reel">

                            <article>
                                <a href="http://mdomaradzki.deviantart.com/art/Bueller-III-351975087"
                                   className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic01.jpg"} alt=""/></a>
                                <header>
                                    <h3><a href="#">Pulvinar sagittis congue</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="http://mdomaradzki.deviantart.com/art/Disco-351602759"
                                   className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic02.jpg"} alt=""/></a>
                                <header>
                                    <h3><a href="#">Fermentum sagittis proin</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic03.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Sed quis rhoncus placerat</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic04.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Ultrices urna sit lobortis</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic05.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Varius magnis sollicitudin</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic01.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Pulvinar sagittis congue</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic02.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Fermentum sagittis proin</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic03.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Sed quis rhoncus placerat</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic04.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Ultrices urna sit lobortis</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                            <article>
                                <a href="#" className="image featured"><img
                                    src={prefix + "/webTemplate/images/pic05.jpg"}
                                    alt=""/></a>
                                <header>
                                    <h3><a href="#">Varius magnis sollicitudin</a></h3>
                                </header>
                                <p>Commodo id natoque malesuada sollicitudin elit suscipit magna.</p>
                            </article>

                        </div>
                    </div>
                    <hr/>
                </div>
            </div>
        )
    }
}

export default UserProfile;
