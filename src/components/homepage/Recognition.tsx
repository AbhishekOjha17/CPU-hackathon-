"use client";

import { useEffect, useLayoutEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { usePopup } from '@/context/PopupContext';

import { useGSAP } from "@gsap/react";

gsap.registerPlugin(ScrollTrigger, useGSAP);

export function RecognitionDesk() {
    const textRef = useRef<HTMLDivElement>(null);
    const { openPopup } = usePopup();

    useGSAP(() => {

        if (!textRef.current) return;
        const letters = textRef.current.querySelectorAll(".spanElement");

        gsap.to(".mainSection", {

            scrollTrigger: {
                trigger: ".mainSection",
                start: "top top",
                end: "100%",
                scrub: true,
                pin: true
            }
        })







        let rtl = gsap.timeline({
            scrollTrigger: {
                trigger: textRef.current,
                start: "top 70%",
                end: "+100%",
                scrub: true,
                // pin: true,
            }
        })


        rtl.fromTo(
            letters,
            { y: 100, opacity: 0 },
            { y: 0, opacity: 1, duration: 0.6, stagger: 0.1 },
            0 // position in timeline
        );


        rtl.fromTo(
            ".revealTextBox",
            { clipPath: "inset(0 0 0 100%)" },
            { clipPath: "inset(0 0 0 0%)" },
            ">"
        )

        rtl.fromTo(".revealTextCards",
            { x: 1400 },
            { x: 0 }, "<+0.2"

        )


    }, []);

    const text = "CRED CAT";

    return (
        <section className="bg-[#E0DFDC] mainSection  min-h-[100vh] flex flex-col items-center justify-center text-white">
            <div
                ref={textRef}
                className="relative  ">


                <div className="  text-[60px] md:text-[120px] lg:text-[160px] 2xl:text-[200px]" >
                    <div className="fontA    flex overflow-hidden font-bold text-[#c2c2c2] ">
                        {text.split("").map((char, i) => (
                            <span key={i} className=" spanElement inline-block opacity-0">
                                {char}
                            </span>
                        ))}
                    </div>


                    <div className="revealTextBox absolute top-0 fontA flex overflow-hidden   font-bold text-[#272727]" style={{ clipPath: "inset(0 0 0 100%)" }} >
                        {text.split("").map((char, i) => (
                            <span key={i} className="inline-block  ">
                                {char}
                            </span>
                        ))}

                    </div>

                </div>


                <div className=" revealTextCards absolute top-0 gap-40 flex " style={{ transform: "translateX(1400px)" }} >
                    <div className="  pt-5 pb-7 lg:py-0 overflow-x-scroll lg:overflow-visible   customScrollbar-z    px-8   justify-start lg:justify-center items-center mt-6 flex gap-20 md:gap-10 2xl:gap-20   md:flex " >

                        <div className=" bg-[#E0DFDC] text-[#272727] flex-shrink-0 relative group scroll-fade-card    transtion-all duration-400  lg:hover:-tra nslate-y-[40px] "
                            onClick={() => openPopup("impact", [
                                "BroadcastPro Manufacturer Awards 2025 - Best in AI Winner",
                                "Plan-itU recognized for transformative {AI-driven automation} in linear broadcast content planning and operations.",
                                "/img/elements/cardimg.png"
                            ])} >
                            <div className="impact-card-wrapper overflow-hidden text-white   glass-card  !rounded-[27px] p-1.5" style={{ backgroundColor: "#134C78" }} >

                                <div className="w-[300px] md:w-[250px] h-[400px] md:h-[64vh] lg:w-[250px] lg:h-[350px] 2xl:w-[300px] 2xl:h-[400px]  impact-card overflow-hidden p-6  flex flex-col justify-between items-start">
                                    <img className="child w-[100%] z-[-1] h-[60%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opaci ty-00" src="/img/elements/cardimg.png" alt="" />

                                    <img className="child transition-opacity duration-400 group-hover:opac ity-0 " src="/img/elements/element (11).svg" alt="" />
                                    <div>
                                        <h2 className="   z-1 mt-4 mb-4 fontA font-regular text-2xl md:text-lg 2xl:text-2xl transition-all duration-700 group-hover:tex t-[#F6F1E6]" >Explainability by Design</h2>
                                        <p className="z-1 font-light text-lg md:text-sm xl:text-lg  poppins mb-3 md-20">Every credit decision includes transparent reasoning across financial, behavioral, and macro signals—making it auditable and regulator-ready. </p>

                                    </div>
                                </div>

                            </div>

                            <div className="glow-circle transition-opacity opacity-0 duration-700 group-hover:opacity-0  absolute top-[-10px] left-[-24%] !w-[130%] !h-[30%] ">

                            </div>

                        </div>

                        <div className=" bg-[#E0DFDC] text-[#272727]  flex-shrink-0 relative group scroll-fade-card    transtion-all duration-400  lg:hover:-tra nslate-y-[40px] "
                            onClick={() => openPopup("impact", [
                                "Powering Sony Pictures Networks India Cloud Journey",
                                "Successfully implemented BroadView in {AWS cloud for SPNI}, setting new standards for enterprise-scale migrations.",
                                "/img/elements/cardimg-sony.png"
                            ])} >
                            <div className="impact-card-wrapper  overflow-hidden text-white  glass-card  !rounded-[27px] p-1.5" style={{ backgroundColor: "#80B556" }} >

                                <div className="w-[300px] md:w-[250px] h-[400px] md:h-[64vh] lg:w-[250px] lg:h-[350px] 2xl:w-[300px] 2xl:h-[400px] impact-card overflow-hidden p-6  flex flex-col justify-between items-start">
                                    <img className="child w-[100%] z-[-1] h-[70%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opacity-00" src="/img/elements/cardimg-sony.png" alt="" />

                                    <img className="child transition-opacity duration-400 group-hover:opac ity-0 " src="/img/elements/purple.svg" alt="" />
                                    <div>
                                        <h2 className=" z-1 mt-8 mb-4 fontA font-regular text-2xl md:text-xl 2xl:text-2xl transition-all duration-700 group-hover:t ext-[#F6F1E6]" >Behavioral Risk Optimization</h2>
                                        <p className="z-1 font-light text-lg md:text-sm 2xl:text-lg poppins">The platform dynamically adjusts risk weighting by analyzing behavioral signals, financial stability, and intent indicators—ensuring balanced, context-aware credit decisions.</p>

                                    </div>
                                </div>

                            </div>

                            <div className="glow-circle transition-opacity opacity-0 duration-700 group-hover:opacity-0  absolute top-[-10px] left-[-24%] !w-[130%] !h-[30%] ">

                            </div>

                        </div>

                        <div className=" bg-[#E0DFDC] text-[#272727] flex-shrink-0 relative group  scroll-fade-card  transtion-all duration-400  lg:hover:-tra nslate-y-[40px] "
                            onClick={() => openPopup("impact",


                                [
                                    "Industry Leadership",
                                    "Featured at NAB 2023, The {Broadcast Bridge}, and {Broadcast & CableSat} for pioneering solutions that redefine media operations.",
                                    "/img/elements/cardimg-industry.png"
                                ]
                            )}

                        >
                            <div className="impact-card-wrapper  overflow-hidden   glass-card  !rounded-[27px] p-1" style={{ backgroundColor: "#027B66" }} >

                                <div className="w-[300px] h-[400px] md:w-[250px] md:h-[64vh] lg:w-[250px] lg:h-[350px] 2xl:w-[300px] 2xl:h-[400px] impact-card overflow-hidden p-6  text-white flex flex-col justify-between items-start">
                                    <img className="child w-[100%] z-[-1] h-[70%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opacity-00   " src="/img/elements/cardimg-industry.png" alt="" />

                                    <img className="child transition-opacity duration-400 group-hover:opa city-0 " src="/img/elements/pink.svg" alt="" />
                                    <div>
                                        <h2 className="   z-1 mt-8 mb-4 fontA font-regular text-2xl md:text-xl 2xl:text-2xl transition-all duration-700 group-hover:t ext-[#F6F1E6]" >Continuously Evolving Risk Model </h2>
                                        <p className="z-1 font-light text-lg md:text-sm 2xl:text-lg poppins">As new applications, conversations, and macro signals emerge, the system continuously refines risk models and improves future decision accuracy.</p>

                                    </div>
                                </div>

                            </div>

                            <div className="glow-circle transition-opacity opacity-0 duration-700 group-hover:opacity-00  absolute top-[-10px] left-[-24%] !w-[130%] !h-[30%] ">

                            </div>

                        </div>



                    </div>
                </div>




            </div>

        </section>
    );
}

 

export function RecognitionMob() {
    const textRef = useRef<HTMLDivElement>(null);
    const sectionRef = useRef<HTMLDivElement>(null);
    const { openPopup } = usePopup();

    useGSAP(() => {

        if (!textRef.current) return;
        const letters = textRef.current.querySelectorAll(".spanElement");

        gsap.to(sectionRef.current, { 
            scrollTrigger: {
                trigger: sectionRef.current,
                
                start: "top 0%",
                end: () => "+=" + 1000,
                scrub: 1,
                pin: true,
                pinSpacing: true, 
            } 
        })


        let rtl = gsap.timeline({
            scrollTrigger: {
                trigger: textRef.current,
                start: "top 80%",
                end: () => "+=" + 500,
                scrub: true, 
                
            }
        })


        rtl.fromTo(
            letters,
            { y: 100, opacity: 0 },
            { y: 0, opacity: 1, duration: 0.6, stagger: 0.1 }
        );


        rtl.fromTo(
            ".revealTextBoxMob",
            { clipPath: "inset(0 0 0 100%)" },
            { clipPath: "inset(0 0 0 0%)" },

        ) 

        return () => {
            rtl.kill(); 
        }; 

    }, []);

    const text = "CREDCAT";

    return (
        <section ref={sectionRef}   className="bg-[#E0DFDC]   h-[100vh]   flex flex-col items-center justify-center text-white">
            <div
                ref={textRef}
                className="relative my-20 ">


                <div className="  text-[60px] md:text-[120px] lg:text-[160px] 2xl:text-[200px]" >
                    <div className="fontA    flex overflow-hidden font-bold text-[#272727] ">
                        {text.split("").map((char, i) => (
                            <span key={i} className=" spanElement inline-block opacity-0">
                                {char}
                            </span>
                        ))}
                    </div>


                    <div className="revealTextBoxMob absolute top-0 fontA flex overflow-hidden   font-bold text-[#c2c2c2]" style={{ clipPath: "inset(0 0 0 100%)" }} >
                        {text.split("").map((char, i) => (
                            <span key={i} className="inline-block  ">
                                {char}
                            </span>
                        ))}

                    </div>

                </div>


                <div className=" revealTextCards mt-5  gap-40 flex "  >
                    <div className=" relative mx-auto items-center flex flex-col gap-5  " >

                        <div className=" bg-[#E0DFDC] absolute top-0 text-[#272727] flex-shrink-0   group scroll-fade-card    transtion-all duration-400  lg:hover:-tra nslate-y-[40px] ">
                            <div className="impact-card-wrapper  overflow-hidden text-white   glass-card  !rounded-[27px] p-1.5" style={{ backgroundColor: "#134C78", height: "fit-content" }} >

                                <div className="impact-card overflow-hidden p-6  flex flex-col justify-between items-start" style={{ height: "fit-content" }} >
                                    <img className="child w-[100%] z-[-1] h-[60%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opaci ty-00" src="/img/elements/cardimg.png" alt="" />

                                    <img className="child hidden transition-opacity duration-400 group-hover:opac ity-0 " src="/img/elements/element (11).svg" alt="" />
                                    <div>
                                        <h2 className="   z-1 mt-0 mb-2 fontA font-regular text-2xl transition-all duration-700 group-hover:tex t-[#F6F1E6]" >Explainability by Design</h2>
                                        <p className="z-1 font-light text-lg poppins">Every AI response is generated with internal reasoning paths mapped across sources, concepts, and relationships—making the system auditable, debuggable, and trustworthy. </p>

                                    </div>
                                </div>

                            </div>
                        </div>

                        <div className=" bg-[#E0DFDC] absolute top-0 text-[#272727] flex-shrink-0   group scroll-fade-card    transtion-all duration-400  lg:hover:-tra nslate-y-[40px] "
                        >
                            <div className="impact-card-wrapper  overflow-hidden text-white  glass-card  !rounded-[27px] p-1.5" style={{ backgroundColor: "#80B556", height: "fit-content" }} >

                                <div className="impact-card overflow-hidden p-6  flex flex-col justify-between items-start" style={{ height: "fit-content" }} >
                                    <img className="child w-[100%] z-[-1] h-[70%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opacity-00" src="/img/elements/cardimg-sony.png" alt="" />

                                    <img className="child transition-opacity hidden duration-400 group-hover:opac ity-0 " src="/img/elements/purple.svg" alt="" />
                                    <div>
                                        <h2 className=" z-1  mb-2 fontA font-regular text-2xl transition-all duration-700 group-hover:t ext-[#F6F1E6]" >Powering Sony Pictures Networks India Cloud Journey</h2>
                                        <p className="z-1 font-light text-lg poppins">Successfully implemented BroadView in  AWS cloud for SPNI , setting new standards for enterprise-scale migrations.</p>

                                    </div>
                                </div>

                            </div>



                        </div>

                        <div className=" bg-[#E0DFDC] absolute top-0 text-[#272727] flex-shrink-0   group  scroll-fade-card  transtion-all duration-400  lg:hover:-tra nslate-y-[40px] " >
                            <div className="impact-card-wrapper  overflow-hidden   glass-card  !rounded-[27px] p-1" style={{ backgroundColor: "#027B66", height: "fit-content" }} >

                                <div className="impact-card overflow-hidden p-6  text-white flex flex-col justify-between items-start " style={{ height: "fit-content" }}  >
                                    <div>
                                        <h2 className="   z-1 mt-8 mb-4 fontA font-regular text-2xl transition-all duration-700 group-hover:t ext-[#F6F1E6]" >Industry <br /> Leadership </h2>
                                        <p className="z-1 font-light text-lg poppins">Featured at NAB 2023, The Broadcast Bridge, and  Broadcast & CableSat   for pioneering solutions that redefine media operations. </p>

                                    </div>
                                </div>

                            </div>

                            <div className="glow-circle transition-opacity opacity-0 duration-700 group-hover:opacity-00  absolute top-[-10px] left-[-24%] !w-[130%] !h-[30%] ">

                            </div>

                        </div>



                    </div>
                </div>




            </div>

        </section>
    );
}



export default function() {


    return (
        <div>

            <div className="hidden lg:block" >
                <RecognitionDesk />

            </div>

            <div className="block md:hidden" >
                {/* <RecognitionMob /> */}
            </div>
        </div>
    )
}