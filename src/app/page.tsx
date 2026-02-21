  "use client";
  import "./homepage.css";

  import { useEffect, useState, useRef, useLayoutEffect } from "react";

  import { usePopup } from "@/context/PopupContext";
  import dynamic from "next/dynamic";

  import CTA from "@/components/CTA";
  import Marquee from "react-fast-marquee";

  import Recognition from "@/components/homepage/Recognition";

  import { useGSAP } from "@gsap/react";
  import gsap from "gsap";


  import { TextPlugin } from "gsap/TextPlugin";
  import { ScrollTrigger } from "gsap/ScrollTrigger";

  gsap.registerPlugin(ScrollTrigger, TextPlugin, useGSAP);

  import Counter from "@/components/Counter";
  import Footer from "@/components/Footer";

  const PartialCylinderShell = dynamic(
    () => import("@/components/cylinderShell"),
    { ssr: false }
  );

  interface ScrollProviderProps {
    children: React.ReactNode;
  }

  interface ExtendedDeviceOptions {
    smooth?: boolean;
    breakpoint: number;
    lerp?: number;
    multiplier?: number;
  }

  // type ImpactItem = [string, string, string];

// const impact: ImpactItem[] = [
//   [
//     "BroadcastPro Manufacturer Awards 2025 - Best in AI Winner",
//     "Plan-itU recognized for transformative AI-driven automation in linear broadcast content planning and operations.",
//     "/img/elements/cardimg.png",
//   ],

//   [
//     "Powering Sony Pictures Networks India Cloud Journey",
//     "Successfully implemented BroadView in AWS cloud for SPNI, setting new standards for enterprise-scale migrations.",
//     "/img/elements/cardimg-sony.png",
//   ],

//   [
//     "Industry Leadership",
//     "Featured at NAB 2023, The Broadcast Bridge, and Broadcast & CableSat for pioneering solutions that redefine media operations.",
//     "/img/elements/cardimg-industry.png",
//   ],
//   ["", "", ""],
// ] as const;


export default function Home() {
  const [popup, setPopup] = useState([true, false]);
  const [modal, setModal] = useState(-1);
  const [popupMeet, setPopupMeet] = useState(false);

  const [width, setWidth] = useState<number | null>(null);


  const textRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLDivElement>(null);

  const [loading, setLoading] = useState(false);

  const { openPopup } = usePopup();

  // const trustedByImages = [
  //   "/img/trustedby/1 (1).png",
  //   "/img/trustedby/1 (2).png",
  //   "/img/trustedby/1 (3).png",
  //   "/img/trustedby/1 (4).png",
  //   "/img/trustedby/1 (5).png",
  //   "/img/trustedby/1 (6).png",
  //   "/img/trustedby/1 (7).png",
  // ];

  useGSAP(() => {
    if (typeof window === "undefined") return;

    const handleResize = () => setWidth(window.innerWidth);

    handleResize();

    window.addEventListener("resize", handleResize);

    const mm = gsap.matchMedia();




    gsap.fromTo(".section-featureDeck",
      {
        opacity: 0,
        // borderTopLeftRadius: "100%",
        // borderTopRightRadius:"100%"
      },

      {
        // borderTopLeftRadius: "0",
        // borderTopRightRadius:"0",

        opacity: 1,
        scrollTrigger: {
          trigger: ".section-featureDeck",
          start: "top 50%",
          end: "top 5%", // pin for 500px of scroll
          pin: false, // element sticks
          scrub: true,
          // markers: true,

          // scroller: "[data-scroll-container]",
        },
      }
    );

    mm.add("(min-width: 1280px)", () => {
      // desktop animations
      gsap.fromTo(".slide-card",
        {
          y: (i) => {
            // console.log("sdsd : ", i);
            return i * 800;
          },
          width: 1080,
        },
        {
          y: (i) => {
            if (i == 3) {
              return i * 85;
            } else {
              return i * 85;
            }
          },

          width: (i) => i * 20 + 1080,
          stagger: 0.3,
          scrollTrigger: {
            trigger: ".slide-card-wrapper",
            start: "top 10%",
            end: "+=2500",
            scrub: true,
            pin: true,
          },
        }
      );

      gsap.fromTo(".section-thatMatter ",

        { opacity: 0.3 },
        {
          opacity: 1,
          scrollTrigger: {
            trigger: ".section-thatMatter ",
            start: "top 40%",
            end: "+=500", // pin for 500px of scroll
            pin: false, // element sticks
            scrub: true,
          },
        }
      );
    });


    mm.add("(max-width: 1280px)", () => {
      gsap.fromTo(".section-thatMatter",

        { opacity: 0.3 },
        {
          opacity: 1,
          scrollTrigger: {
            trigger: ".section-thatMatter",
            start: "top 40%",
            pinSpacing: true,
            end: "+=100",
            scrub: true,
          },
        }
      );

    });

    mm.add("(max-width: 1279px) and (min-width: 1024px)", () => {
      // mobile animations
      gsap.fromTo(
        ".slide-card",
        {
          y: (i) => i * 1100,
          width: 1000,
        },
        {
          y: (i) => {
            if (i == 3) {
              return i * 80;
            }
            return i * 80;
          },
          width: (i) => i * 10 + 1000,
          stagger: 0.15,
          scrollTrigger: {
            trigger: ".slide-card-wrapper",
            start: "top 10%",
            end: "+=1500",
            scrub: true,
            pin: true,
          },
        }
      );
    });

    mm.add("(max-width: 1023px) and (min-width: 768px)", () => {
      // mobile animations
      gsap.fromTo(
        ".slide-card",
        {
          y: (i) => i * 1100,
          width: 740,
        },
        {
          y: (i) => {
            if (i == 3) {
              return i * 80;
            }
            return i * 80;
          },
          width: (i) => i * 10 + 740,
          stagger: 0.15,
          scrollTrigger: {
            trigger: ".slide-card-wrapper",
            start: "top 10%",
            end: "+=1500",
            scrub: true,
            pin: true,
          },
        }
      );
    });

    mm.add("(max-width: 767px) and (min-width: 640px)", () => {
      // mobile animations
      gsap.fromTo(
        ".slide-card",
        {
          y: (i) => i * 1100,
          // width: 580
        },
        {
          y: (i) => {
            if (i == 3) {
              return i * 80;
            }
            return i * 80;
          },
          // width: (i) => i * 10 + 580,
          stagger: 0.15,
          scrollTrigger: {
            trigger: ".slide-card-wrapper",
            start: "top 10%",
            end: "+=1500",
            scrub: true,
            pin: true,
          },
        }
      );
    });

    mm.add("(max-width: 640px)", () => {
      // mobile animations
      gsap.fromTo(
        ".slide-card",
        {
          y: (i) => i * 600,
          // width: 360
        },
        {
          // y: (i) => { if (i == 3) { return i * 80 + 30 }; return i * 80 },
          y: (i) => i * 0,

          // width: (i) => i * 0 + 360,
          stagger: 0.3,
          scrollTrigger: {
            trigger: ".slide-card-wrapper",
            start: "top 10%",
            end: "+=1500",
            scrub: true,
            pin: true,
          },
        }
      );

      // gsap.fromTo(
      //   ".sec-awards-card-mob",
      //   {
      //     y: (i) => {
      //       if (i == 0) {
      //         return 0;
      //       }
      //       return i * 600;
      //     },
      //   },
      //   {
      //     // y: (i) => { if (i == 3) { return i * 80 + 30 }; return i * 80 },
      //     y: (i) => i * 0,

      //     stagger: 0.3,
      //     scrollTrigger: {
      //       trigger: ".sec-awards",
      //       start: "top top",
      //       end: "+=1500",
      //       scrub: true,
      //       pin: true,
      //     },
      //   }
      // );
    });

    const slideCards = document.querySelectorAll(".slide-card");

    slideCards.forEach((card, index) => {
      gsap.fromTo(
        card.querySelector(".slide-card-h2"),
        { opacity: 0 },
        {
          opacity: 1,

          scrollTrigger: {
            trigger: card,
            start: index == 2 ? "700px" : "800px",
            end: index == 2 ? "900px" : "1700px",
            scrub: true,
          },
        }
      );
    });

    gsap.from(".section-shell", {
      opacity: 0,
    });

    gsap.to(".section-shell", {
      opacity: 1,
      scrollTrigger: {
        trigger: ".section-shell",
        start: "top 80%",
        end: "top top", // pin for 500px of scroll
        pin: false, // element sticks
        scrub: true,
      },
    });


    gsap.fromTo(".section-xslide ",

      { opacity: 0.3 },
      {
        opacity: 1,
        scrollTrigger: {
          trigger: ".section-xslide ",
          start: "top 40%",
          end: "+=500", // pin for 500px of scroll
          pin: false, // element sticks
          scrub: true,
        },
      }
    );




    const track = document.querySelector(
      ".trustedBy-logo"
    ) as HTMLElement | null;

    const trackWidth = track?.offsetWidth;

    // gsap.to(".trustedBy-logo", {
    //   x: `-=${trackWidth}`, // shift by one full track
    //   duration: 40,
    //   ease: "linear",
    //   repeat: -1,
    //   modifiers: {
    //     x: gsap.utils.unitize((x) => parseFloat(x) % -trackWidth!), // reset smoothly
    //   },
    // });









    gsap.fromTo(".home-reveal",
      {
        clipPath: "circle(0% at 50% 50%)", // start completely clipped
      },
      {
        clipPath: "circle(80%  at 50% 50%)", // expand until it covers screen
        duration: 1,
        ease: "circ.inOut",
      }
    );

    gsap.to(".reveal-text-A span", {
      y: "0%",
      opacity: 1,
      duration: 2.5,
      ease: "circ.inOut",
    });

    gsap.to(".reveal-text-B span", {
      delay: 1.9,

      y: "0%",
      opacity: 1,
      duration: 0.7,
      ease: "power2.out",
    });


    gsap.fromTo(".modal",
      { right: -400 },
      {
        delay: 2.7,
        right: 30,

        duration: 0.4,
        ease: "power4.out",
      }
    );




    // gsap.fromTo(".anyClassWrap", { opacity: 0, x: 0 }, {

    //   opacity: 1, x: 200,
    //   scrollTrigger: {
    //     trigger: ".anyClassWrap",
    //     start: "top top",
    //     end: () => "+=" + 1000, scrub: 1, pin: true,
    //     pinSpacing: true,
    //   }
    // }) 

    if (!textRef.current) return;
    const letters = textRef.current.querySelectorAll(".spanElement");


    gsap.to( sectionRef.current, {
      scrollTrigger: {
        trigger: sectionRef.current,

        start: "top 0%",
        end: () => "+=" + 1400,
        scrub: 1,
        pin: true,
        pinSpacing: true,
      }
    })

    let rtll = gsap.timeline(  {
      scrollTrigger: {
        trigger: sectionRef.current,

        start: "top 0%",
        end: () => "+=" + 400,
        scrub: 1, 
      }
    })

    rtll.fromTo(  
      letters,
      { y: 100, opacity: 0 },
      { y: 0, opacity: 1,   stagger: 0.1  }
    );


    rtll.fromTo(
      ".revealTextBoxMob",
      { clipPath: "inset(0 0 0 100%)" },
      { clipPath: "inset(0 0 0 0%)"  },

    )

    const recogCardMob = gsap.utils.toArray(".scroll-fade-card");

    let rtl = gsap.timeline({
      scrollTrigger: {
        trigger: textRef.current,
        start: "top 80%",
        end: () => "+=" + 1400,
        scrub: true,

      }
    }) 

    gsap.set(recogCardMob, { y: 800 });
    
    rtl.fromTo(
      recogCardMob, { y:   800  }, { y:  0, stagger: 2 } 
    )



    if (window.innerWidth >= 1024) {

      const screenWidth = window.screen.width;
      let textScrollOffset = screenWidth;

      let tlm = gsap.timeline({
        scrollTrigger: {
          trigger: ".textScroll-cont",
          start: "top 0%",
          end: "+300%",
          scrub: true,
          markers: false,
          pin: true
        },
      });

      tlm.fromTo(
        ".textScroll-01",
        {
          x: -textScrollOffset,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-01b",
        {
          x: textScrollOffset,
        },
        {
          x: 300,
        }
      );

      tlm.fromTo(
        ".textScroll-02",
        {
          x: -textScrollOffset-200,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-03",
        {
          x: -textScrollOffset/2,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-04",
        {
          x: textScrollOffset+200,
        },
        {
          x: 200,
        }
      );

      tlm.fromTo(
        ".textScroll-05",
        {
          x: -textScrollOffset-200,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-06",
        {
          x: textScrollOffset + 400,
        },
        {
          x: 0,
        }
      );

      let tl = gsap.timeline({
        scrollTrigger: {
          trigger: ".section-featureDeck",
          start: "top bottom",
          end: "top 30%",
          scrub: true,
        },
      });

      tl.fromTo(
        ".u-element",
        { scale: 25, opacity: 1 },
        { scale: 2, opacity: 1, duration: 0.5 }
      ).to(".u-element", { delay: 0.3, y: -200, duration: 0.5 });

      let tly = gsap.timeline({
        scrollTrigger: {
          trigger: ".section-xslide",
          start: "top bottom",
          end: "top 30%",
          scrub: true,
        },
      });

      tly
        .fromTo(
          ".o-element",
          { scale: 50, opacity: 1 },
          { scale: 2, opacity: 1, duration: 0.5 }
        )
        .to(".o-element", { delay: 0.3, opacity: 0, y: -100, duration: 0.5 });
    }


    if (window.innerWidth <= 1023) {

      let textScrollOffset = 1500;

      let tlm = gsap.timeline({
        scrollTrigger: {
          trigger: ".textScroll-cont",
          start: "top 0%",
          end: "+200%",
          scrub: true,
          markers: false,
          pin: true,
          pinSpacing: true,
        },
      });

      tlm.fromTo(
        ".textScroll-01",
        {
          x: -textScrollOffset,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-01b",
        {
          x: textScrollOffset,
        },
        {
          x: 50,
        }
      );

      tlm.fromTo(
        ".textScroll-02",
        {
          x: -textScrollOffset,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-03",
        {
          x: -800,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-04",
        {
          x: textScrollOffset,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-05",
        {
          x: -textScrollOffset - 200,
        },
        {
          x: 0,
        }
      );

      tlm.fromTo(
        ".textScroll-06",
        {
          x: textScrollOffset + 200,
        },
        {
          x: 0,
        }
      );
    }

    gsap.fromTo(".thatMatter-card-LU ",
      { x: "-100%", opacity: 0 },
      {
        x: 0,
        opacity: 1,

        scrollTrigger: {
          trigger: ".thatMatter-card-LU",
          start: "top bottom",
          end: "bottom bottom",
          scrub: true,
          // scroller: "[data-scroll-container]",
        },
      }
    );

    gsap.fromTo(".thatMatter-card-LD ",
      { x: "-100%", opacity: 0 },
      {
        x: 0,
        opacity: 1,

        scrollTrigger: {
          trigger: ".thatMatter-card-LD",
          start: "top bottom",
          end: "bottom bottom",
          scrub: true,
          // scroller: "[data-scroll-container]",
        },
      }
    );

    gsap.fromTo(".thatMatter-card-RU ",
      { x: "+10%", opacity: 0 },
      {
        x: 0,
        opacity: 1,

        scrollTrigger: {
          trigger: ".thatMatter-card-RU",
          start: "top bottom",
          end: "bottom bottom",
          scrub: true,
          // scroller: "[data-scroll-container]",
        },
      }
    );

    gsap.fromTo(".thatMatter-card-RD ",
      { x: "+10%", opacity: 0 },
      {
        x: 0,
        opacity: 1,

        scrollTrigger: {
          trigger: ".thatMatter-card-RD",
          start: "top bottom",
          end: "bottom bottom",
          scrub: true,
        },
      }
    );



    gsap.to(".reveal-text-1 span", {
      y: "0%",
      opacity: 1,
      duration: 1,
      ease: "power4.out",
      stagger: 0.2, // animate each line one after another
      scrollTrigger: {
        trigger: ".reveal-text-1",
        start: "top 70%", // start when text enters viewport
        toggleActions: "play none none reverse",

        // scroller: "[data-scroll-container]",
      },
    });

    gsap.to(".reveal-text-2 span", {
      y: "0%",
      opacity: 1,
      duration: 1,
      ease: "power4.out",
      stagger: 0.2, // animate each line one after another
      scrollTrigger: {
        trigger: ".section-featureDeck",
        start: "top 70%", // start when text enters viewport
        // toggleActions: "play none none reverse",
        // scrub: true,
        toggleActions: "play reset play reset",
        // scroller: "[data-scroll-container]",
      },
    });

    gsap.to(".reveal-text-3 span", {
      y: "0%",
      opacity: 1,
      duration: 1,
      ease: "power4.out",
      stagger: 0.2, // animate each line one after another
      scrollTrigger: {
        trigger: ".reveal-text-3",
        start: "top 50%", // start when text enters viewport prev: top 80
        // toggleActions: "play none none reverse",
        // scrub: true,
        end: "top 5%",
        toggleActions: "play none play reset",

        // scroller: "[data-scroll-container]",
      },
    });

    gsap.to(".reveal-text-4 span", {
      y: "0%",
      opacity: 1,
      duration: 1,
      ease: "power4.out",
      stagger: 0.2, // animate each line one after another
      scrollTrigger: {
        trigger: ".reveal-text-4",
        start: "top 70%", // start when text enters viewport
        // toggleActions: "play none none reverse",
        // scrub: true,
        toggleActions: "play none play reset",

        // scroller: "[data-scroll-container]",
      },
    });

    gsap.fromTo(".popup-gsap",
      { scale: 1 }, // starting state
      {
        scale: 1.2,
        duration: 0.3,
        ease: "back.out(2)",
        yoyo: true,
        repeat: 1, // goes forward, then back
      }
    );

    gsap.fromTo(".slidedown-gsap",
      { y: -50, opacity: 0 }, // start 100px down
      {
        y: 0,
        opacity: 1,
        duration: 1.2,
        ease: "power3.out", // smooth easing
      }
    );

    gsap.fromTo(".slideup-gsap",
      { y: 50, opacity: 0 }, // start 100px down
      {
        y: 0,
        opacity: 1,
        duration: 1.2,
        ease: "power3.out", // smooth easing
      }
    );

    gsap.fromTo(".slideup-gsap-scroll",
      { y: 50, opacity: 0 }, // start 100px down
      {
        y: 0,
        opacity: 1,
        duration: 1.2,
        ease: "power3.out", // smooth easing

        scrollTrigger: {
          trigger: ".slideup-gsap-scroll",
          start: "top 80%", // when element enters viewport
          toggleActions: "play none none reverse",
          // scroller: "[data-scroll-container]",
        },
      }
    );

    gsap.fromTo(".fadein-scroll-gsap",
      { opacity: 1 },
      {
        opacity: 1,
        duration: 1.2,
        ease: "power2.out",
        scrollTrigger: {
          trigger: ".box",
          start: "top 80%", // when element enters viewport
          toggleActions: "play none none reverse",
          // scroller: "[data-scroll-container]",
        },
      }
    );

    gsap.fromTo(".fadein-scroll-gsap-z",
      { opacity: 0 },
      {
        opacity: 1,
        duration: 1.2,
        ease: "power2.out",
        scrollTrigger: {
          trigger: ".fadein-scroll-gsap-z",
          start: "top 70%", // when element enters viewport
          toggleActions: "play none none reverse",
          // scroller: "[data-scroll-container]",
          scrub: true,
        },
      }
    );

    gsap.fromTo(".fadein-gsap",
      { opacity: 0 },
      {
        opacity: 1,
        duration: 1.2,
        ease: "power2.out",
      }
    );

    gsap.fromTo(".fadein-gsap-delay1",
      { opacity: 0 },
      {
        opacity: 1,
        duration: 1.2,
        ease: "power2.out",
      }
    );

    gsap.utils.toArray<HTMLElement>(".scroll-fade").forEach((el) => {
      gsap.fromTo(
        el,
        { opacity: 0, y: 0 },
        {
          opacity: 1,
          y: 0,
          duration: 1.5,
          ease: "power3.out",
          scrollTrigger: {
            trigger: el,
            start: "top 80%",
            toggleActions: "play none none reverse",
            // scroller: "[data-scroll-container]",
            scrub: true,
          },
        }
      );
    });

    gsap.fromTo(".scroll-fade-meet",
      { opacity: 0, y: 0 },
      {
        opacity: 1,
        y: 0,
        duration: 1.5,
        ease: "power3.out",
        scrollTrigger: {
          trigger: ".scroll-fade-meet",
          start: "top bottom",
          end: "top 80%",
          toggleActions: "play none none reverse",
          // scroller: "[data-scroll-container]",
          scrub: true,
        },
      }
    );

    gsap.fromTo(".scroll-fade-s2",
      { opacity: 0, y: 10 },
      {
        opacity: 1,
        y: 0,
        delay: 0.2,
        ease: "power4.out",

        duration: 0.5,
        scrollTrigger: {
          trigger: ".scroll-fade-s2",
          start: "top 70%",
          end: "top 20%",
          toggleActions: "restart none restart none",
        },
      }
    );

    gsap.fromTo(".scroll-fade-s3",
      { opacity: 0, y: 10 },
      {
        opacity: 0.6,
        y: 0,
        delay: 0.1,
        ease: "power4.out",

        duration: 0.6,
        scrollTrigger: {
          trigger: ".scroll-fade-s3",
          start: "top 80%",
          end: "top 10%",
          toggleActions: "play none play reset",
        },
      }
    );


    gsap.fromTo(".stripSlide",
      {
        bottom: -1150,
        height: 1100,
        ease: "bounce.inOut"
      },
      {
        bottom: 0,
        height: 700,

        scrollTrigger: {
          trigger: ".heroCont", // can also use a wrapper container
          start: "top 10px",
          scrub: true,
        },
      }
    );






    const counter = document.getElementById("counter");
    if (!counter) return;

    let i = 0;
    let interval: any;
    const countUpto = 250; // your target
    const duration = 2000; // total time in ms

    function startCount() {
      i = 0;
      interval = setInterval(() => {
        if (i <= countUpto) {
          counter!.innerText = i + "+";
          i++;
        } else {
          clearInterval(interval);
        }
      }, duration / countUpto);
    }

    // Run only when visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            startCount();
            observer.unobserve(counter); // 👈 run only once
          }
        });
      },
      { threshold: 0.5 } // 50% visible
    );

    observer.observe(counter);
    return () => window.removeEventListener("resize", handleResize);





  }, []);


  const text = "Recognition";




  if (!loading) {
    return (
      <div className="relative bg-[#272727]">
        {/* <Popup type={false} isOpen={ibc} onClose={() => setIbc(false)} />  */}
        <div
          className={`${modal !== -1 ? "" : "hidden"
            } fixed flex flex-col z-[4] bg-[#d9d9d6] w-screen justify-center h-[100vh] items-center`}
        >
          {modal !== -1 && (
            <>
              <div className="relative mx-auto group w-7xl cursor-pointer transition-all duration-400 pointer-events-none">
                <div className="impact-card-wrapper glass-card rounded-[27px] p-1.5">
                  <div className="!w-[100%] text-[#272727] !h-fit impact-card-modal overflow-hidden flex justify-between items-center">
                    {/* Left side */}
                    <div className="px-20 py-4 w-[40%]">
                      <h2 className="z-10 mb-4 fontA font-bold text-4xl transition-all duration-700">
                        {impact[modal][0]}
                      </h2>
                      <p className="z-10 mt-5 opacity-70 text-2xl poppins">
                        {impact[modal][1]}
                      </p>
                    </div>

                    {/* Right side */}
                    <div className="w-[50%] overflow-hidden">
                      <img
                        className="z-[-1] !w-full fadeout-mask-left transition-opacity opacity-70 duration-700"
                        src={impact[modal][2]}
                        alt="Impact card"
                      />
                    </div>
                  </div>
                </div>

                {/* Glow circle */}
                <div className="glow-circle transition-opacity opacity-30 duration-700 absolute top-0 left-0 !w-[70%] !h-[30%]" />
              </div>

              {/* Close button */}
              <p
                className="mt-5 cursor-pointer poppins text-underline border-b border-dash"
                onClick={() => setModal(-1)}
              >
                <img
                  className="w-[30px]"
                  src="https://img.icons8.com/ios-filled/50/multiply.png"
                  alt="Close"
                />
              </p>
            </>
          )}
        </div>

        <div className="relative top-0 overflow-hidden">
          <div className="bg-black ">
            <div className="relative text-white">
              <div className=" absolute w-screen h-screen flex items-center justify-center">
                <div
                  className={`  ${popup[1] == true ? "" : "hidden"
                    } glass-card  mx-auto !rounded-[20px] overflow-hidden top-0 right-0 z-[2] text-white  flex flex-col justify-center px-3 py-5 md:px-10 md:py-15 h-fit`}
                  style={{
                    backdropFilter: "blur(50px)",
                    borderRadius: "0px",
                    borderWidth: "0px",
                    background:
                      "linear-gradient(135deg, rgb(162 89 255 / 0%) 0%, rgba(255, 31, 125, 0.32) 100%)",
                  }}
                >
                  <div className="absolute top-5 right-5">
                    <div
                      onClick={() => {
                        setPopup((prev) => [prev[0], !prev[1]]);
                      }}
                      className="glass-card text-sm !rounded-[15px] opacity-70 cursor-pointer z-1 flex items-center py-4 px-4 "
                      style={{
                        boxShadow:
                          "0 8px 32px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgb(255 255 255 / 0%), inset 0 -1px 0 rgba(255, 255, 255, 0.1), inset 0 0 0px 0px rgba(255, 255, 255, 0)",
                      }}
                    >
                      <span>
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 16 16"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M0.353516 0.646484L14.6595 14.9524M15.1718 0.646484L0.865855 14.9524"
                            stroke="white"
                          />
                        </svg>
                      </span>
                    </div>
                  </div>

                  <div className="">
                    <div className=" ">
                      <h2 className="z-[2] mt-8 mb-4 fontA font-bold text-3xl  ">
                        Meet UTO at IBC!
                      </h2>

                      <p className="fontA font-regular text-2xl">
                        {" "}
                        Hall 2, Stand B45{" "}
                      </p>
                    </div>

                    <div className="poppins font-light my-8 flex flex-col gap-3 text-light">
                      <div className="flex justify-between flex-wrap  gap-3">
                        <input
                          className="py-4 px-5 bg-[#ffffff0d] rounded-[20px] "
                          type="text"
                          placeholder="First Name"
                        />
                        <input
                          className="py-4 px-5 bg-[#ffffff0d] rounded-[20px] "
                          type="text"
                          placeholder="Last Name"
                        />
                      </div>

                      <input
                        className="py-4 px-5 bg-[#ffffff0d] rounded-[20px] "
                        type="text"
                        placeholder="Email"
                      />
                      <input
                        className="py-4 px-5 bg-[#ffffff0d] rounded-[20px] "
                        type="text"
                        placeholder="Contact Number"
                      />
                    </div>

                    <div
                      className="mt-5 CTABUTTON flex relative overflow-hidden rounded-[20px]"
                      onClick={() => {
                        setPopupMeet(true);
                      }}
                    >
                      <div className="glass-card fontA  cursor-pointer z-1 flex items-center py-5 px-6  ">
                        <span className="opacity-70 ">BOOK A MEETING </span>
                      </div>

                      <div
                        className="!absolute top-[-15px]  glow-circle opacity-30  z-0 !w-[150px] !h-[80px]"
                        style={{ borderRadius: "0px" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>



              {/* home-reveal */}
              <section data-speed="1" className="heroCont flex items-center bg-[#001F1E] overflow-hidden h-[100vh] w-[100vw] justify-center">
                <div className="overflow-hidden flex items-center h-[100%] w-[100%] ">
                  <section
                    className={`hero flex relative z-[2] flex-col justify-center relative  h-[100vh] w-[100%] `}
                  >
                    <div className="stripSlide hidden h-[300px] absolute left-0 z-1 bottom-0 flex justify-between flex-col">

                      {[1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6, 8, 8, 200].map((item, index) => (
                        <div key={index}
                          // 8,8, 6,5,4,3,2,2,2,2, 1,1,1
                          // 2b7578 1f1f1f
                          className={`w-[100vw] bg-[#161616]`}
                          style={{ height: `${3 * item}px` }}
                        ></div>
                      ))}
                    </div>

                    <div className="absolute h-[100v] w-[100vw]">
                      <div className="relative h-screen w-full overflow-hidden bg-[#272727]">
                        {/* Background Video */}
                        <video
                          className="absolute top-1/2 left-1/2 min-w-full min-h-full w-auto h-auto -translate-x-1/2 -translate-y-1/2 object-cover "
                          autoPlay
                          muted
                          loop
                          playsInline
                        >
                          <source src="/videos/banner2.mp4" type="video/mp4" />
                          Your browser does not support the video tag.
                        </video>

                        {/* Optional dark overlay */}
                        <div className="absolute inset-0 bg-black/40" />

                        {/* Foreground Content */}
                        <div className="relative z-10 flex flex-col items-center justify-center h-full text-white"></div>
                      </div>
                    </div>

                    <div className=" hidden modal-cont z-[10] absolute bottom-2 right-0">
                      <div
                        className={`${popup[0] == true ? "" : "hidden"
                          } modal items-start !absolute  bottom-5 right-5 z-2 rounded-[25px] flex flex-col items-end w-fit p-2 md:p-3 xl:p-4 `}
                      >
                        <div className="hover-card-gradient !h-[120px] lg:!h-[178px] text-white p-2 lg:p-2 xl:p-5   ">
                          <div className="flex justify-between">
                            <img
                              className=" absolute right-[10px] sm:left-0 sm:static    lg:block   w-[40px]"
                              src="/img/elements/element (12).webp"
                              alt=""
                            />
                            {/* <button onClick={() => { setPopup(prev => [!prev[0], prev[1]]) }} className="cursor-pointer fontA font-light" >   close  </button> */}
                          </div>
                          {/* {idk what is was doing removed } */}
                          {/* <h2 className="fontA lg:mt-2 font-bold text-xl lg:text-2xl">
                            IBC 2025
                          </h2>
                          <p className="poppins">
                            {" "}
                            12<sup>th</sup> to 15<sup>th</sup> September <br />{" "}
                            Meet UTO at <br /> <b>Hall 2, Stand B45</b> <br />{" "}
                          </p> */}
                        </div>

                        <div className="flex w-[100%] mt-1 md:mt-4 justify-between  items-center ">
                          <div
                            className="relative cursor-pointer "
                            onClick={() => openPopup("default")}
                          >
                            <button className=" w-fit bg-[#c4bba82b] p-2 lg:p-3 rounded-[11px] text-white fontA font-medium   ">
                              {" "}
                              BOOK A MEETING{" "}
                            </button>
                            <div className="absolute top-[-50%] gradient-fill-A"></div>
                          </div>
                          {/* <img className="  w-[40px] md:hidden" src="/img/elements/element (12).webp" alt="" /> */}
                        </div>
                      </div>
                    </div>

                    <div className="relative ">
                      <div className="  flex flex-col items-center">
                        <h1 className="  mt-0  mt-[-80px] sm:mt-0 lg:mt-[-120px]   xl:mt-[0px] z-2 fontA font-medium text-center reveal-text-A px-10 text-5xl sm:text-6xl lg:text-7xl lg:leading-25 ">
                          <div className="hidden lg:flex  gap-5 ">
                            <span> Credit Decisions </span>
                            <span className="text-[#3AFF16]"> Explained</span>
                          </div>

                          <div className=" lg:hidden">
                            <span> Engineered For </span>
                            <span className="text-[#3AFF16]"> Impact </span>
                          </div>
                        </h1>

                        <p className="poppins font-light z-2 text-xl sm:text-2xl text-center mt-5 px-8 sm:px-0  reveal-text-B max-w-3xl px-1">
                          {/* <span>
                            {" "}
                            Transforming media complexity into competitive
                            advantage <br /> Advanced tech ● Proven results
                            <br className="md:hidden" /> ● Unstoppable growth
                          </span> */}
                          {/* <span>
                            Turning scattered notes into structured knowledge <br />
                            AI-powered graphs ● Faster understanding
                            <br className="md:hidden" /> ● Smarter learning
                          </span> */}
                          <span>
                          AI-powered credit intelligence combining predictive scoring, behavioral analysis, and macro-risk simulation — built for real-world finance.
                          </span>

                        </p>
                      </div>
                    </div>
                  </section>
                </div>
              </section>
              
              {/*Marquee sections slider */}
              {/* <section className="bg-[#282828] z-[2] section-clientlogo relative py-10  ">
                <div className="max-w-7xl mx-auto px-10 flex flex-col items-center md:flex-row  md:justify-left md:items-center">
                  <div>
                    <h2 className="text-center lg:text-left lg:w-[240px] fontA text-white text-xl md:text-2xl mr-5   ">
                      Trusted by industry’s <br className=" hidden md:block" />{" "}
                      boldest
                    </h2>
                  </div>

                  <div className="flex   mt-6 lg:mt-0 overflow-hidden">
                    <div className="trustedBy-logo shrink-0 mr-8 flex   items-center gap-8 md:gap-12">
                      {Array.from({ length: 12 }).map((_, i) => (

 
                      (  <img
                          key={i}
                          src={`/img/trustedby/1 (${i + 1}).png`}
                          alt={`Logo ${i + 1}`}
                          className="h-10   object-contain opacity-90 hover:opacity-100 transition"
                        />)
                      ))}

                      {Array.from({ length: 14 }).map((_, i) => (
                        <img
                          key={i}
                          src={`/img/trustedby/1 (${i + 1}).png`}
                          alt={`Logo ${i + 1}`}
                          className="h-10   object-contain opacity-90 hover:opacity-100 transition"
                        />
                      ))}
                    </div>

                    <div
                      className="trustedBy-logo   flex   items-center gap-8 md:gap-12"
                      aria-hidden="true"
                    >
                      {Array.from({ length: 14 }).map((_, i) => (
                        <img
                          key={i}
                          src={`/img/trustedby/1 (${i + 1}).png`}
                          alt={`Logo ${i + 1}`}
                          className="h-10   object-contain opacity-90 hover:opacity-100 transition"
                        />
                      ))}

                      {Array.from({ length: 14 }).map((_, i) => (
                        <img
                          key={i}
                          src={`/img/trustedby/1 (${i + 1}).png`}
                          alt={`Logo ${i + 1}`}
                          className="h-10   object-contain opacity-90 hover:opacity-100 transition"
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </section> */}

              <section className="bg-[#F6F1E7]   z-[1] text-[#3b3b3b] relative section-shell opacity-0 "
                style={{
                  backgroundImage: "url(img/homeBanner.jpg)",
                  backgroundSize: "cover",
                }}
              >


                <div className="absolute hidden h-[100v] w-[100vw]">
                  <div className="relative h-screen w-full overflow-hidden bg-[#F6F1E7]">
                    {/* Background Video */}
                    <video
                      className="absolute top-1/2 left-1/2 min-w-full min-h-full w-auto h-auto -translate-x-1/2 -translate-y-1/2 object-cover "
                      autoPlay
                      muted
                      loop
                      playsInline
                    >
                      <source src="/videos/homepage.mp4" type="video/mp4" />
                      Your browser does not support the video tag.
                    </video>


                  </div>
                </div>

                <div className=" z-20 max-w-7xl mx-auto px-10 ">
                  <div className=" py-10">
                    <div className="    flex flex-col lg:flex-row lg:justify-between lg:items-center  py-10 ">
                      <div className=" lg:w-[50%] ">
                        <h1 className="fontA font-medium   text-4xl sm:text-6xl">
                          <div className="!hidden  mt-10  reveal-text reveal-text-1  lg:!block">
                            <span>Where Credit Decisions</span>
                            <span>
                              Meet {" "}
                              <span className="text-[#02816B]"> Transparency</span>{" "}
                            </span>
                          </div>

                          <div className="reveal-text reveal-text-1   lg:!hidden">
                            <span>
                              Where Students <br className="sm:hidden" /> Meets The{" "}
                              <span className="text-[#02816B]"> Future </span>{" "}
                              <br className="sm:hidden" /> Of Learning
                            </span>

                            <img
                              className="hidden ml-[70vw] sm:ml-[87vw] top-0 absolute"
                              src="/img/cylinder-mob.png"
                              alt=""
                            />
                          </div>
                        </h1>

                        <div className="scroll-fade-s2">
                          <p className="poppins font-light text-xl text-[#888580] mt-4 md:mt-10 ">
                            {/* Media never stands still. Platforms multiply,
                            audiences fragment, and business needs shift faster
                            than ever. */}
                            {/* <span>Modern learning is</span> <span className="poppins font-dark text-xxl text-[#888670]">fragmented</span>. PDFs, notes, slides, and web content live in different places, 
                            making it hard to see connections or recall key ideas. 
                            NoteGraph solves this by structuring your material into 
                            an AI-powered knowledge graph, helping you navigate concepts visually,
                             retrieve context fast, and learn more efficiently. */} 




                            <span>Modern credit scoring is</span>{" "}
                            <span className="poppins font-dark text-xxl text-[#888670]">opaque.</span>.
                            <br />

                            <span>Black-box approvals.</span>
                            <br />

                            <span>No behavioral intelligence.</span>
                            <br />

                            <span>Hidden risk signals.</span>
                            <br /><br />

                            <span>We bring explainable intelligence.</span>
                            <br />

                            <span>AI-powered risk modeling.</span>
                            <br />

                            <span>Clear decisions.</span>{" "}
                            <span>Real accountability.</span>
                          </p>

                          {/* <p className="poppins font-light text-xl text-[#888580] mt-2 md:mt-6 ">
                            With custom learning style customisations, adaptive technology, and an
                            approach that simplifies complexity, we enable our
                            Students to consistently outperform their potential and achieve their goals.
                          </p> */}
                        </div>

                        <div
                          // onClick={() => openPopup("meeting")}
                          className=" CTABUTTON scroll-fade-meet mt-6 relative md:mt-10"
                        >
                          <div
                            className="poppins glass-card cursor-pointer !rounded-[16px] w-[160px] h-[53.6px]  z-2  flex justify-center items-center"
                            style={{
                              backgroundColor: "rgb(3 , 255, 64 , 32%)",
                              border: "1px solid rgb(82 255 89)",
                            }}
                          >
                            <span>Create Account </span>
                          </div>

                          <div className="!absolute top-[-10px] gradient-overlay opacity-60  z-1 !w-[200px] !h-[100px]"></div>
                        </div>
                      </div>

                      <div className="hidden lg:hidden justify-center mt-10 lg:mt-0">
                        <div className="  -rotate-40">
                          <PartialCylinderShell size={500} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="hidden lg:hidden u-element z-[1] h-[80px] md:h-[100px] md:scale-1"></div>
              </section>

              <section className="overflow-hidden  section-featureDeck bg-[#282828] opacity-0 relative text-[#F1ECE2] py-20  "
                style={{}}
              >
                <div className="">
                  <h1 className="fontA font-medium text-center text-4xl sm:text-6xl">
                    <div className=" reveal-text  reveal-text-2 mt-10 !hidden lg:!flex lg:flex-col">
                      <span>Applying AI To</span>
                      <span>
                        {" "}
                        <span className="text-[#8DC55D]">Optimize</span> Credit Decisions 
                      </span>
                    </div>
                    {/* mobile view */}
                    <div className=" reveal-text sm:w-[600px] mx-auto px-2 reveal-text-2 lg:!hidden">
                      <span>
                        Adapting Tech to{" "}
                        <span className="text-[#8DC55D]">Optimize</span> Learning 
                      </span>
                    </div>
                  </h1>
                  
                  {/* {remove cause of no use } */}
                   {/* <p className="scroll-fade-s3 poppins font-light hidden  text-center text-xl mt-4 md:mt-10">
                    Complete media value chain enablement{" "}
                    <span className="font-light">
                      {" "}
                      <br />
                      UTO does it all {" "}
                    </span>
                  </p> */}

                  <div className=" mt-20   ">
                    <div className="max-w-7xl  mt-5 flex flec-col justify-center  slide-card-wrapper relative  mx-auto">
                      <div className="slide-huhh  w-[95vw] xl:w-[1080px] xl:h-[448px] absolute slide-card flex flex-col lg:flex-row lg:justify-between lg:items-center  mx-auto text-white px-5 "
                        style={{ backgroundColor: "rgb(124 172 83)" }}
                      >
                        <div className=" lg:w-[60%] p-2 py-5 lg:p-5 ">

                          <div className="" >
                            <div className="flex items-center gap-5   fontA">
                              <img src="./img/elements/deck-01.png" alt="" />
                              {/* slide-card-h2 */}
                              <h2
                                className="    font-light  text-2xl   "
                                style={{ lineHeight: "normal" }}
                              >
                                {" "}
                                Intelligent <br className="sm:hidden" /> Risk Workflows 
                              </h2>
                            </div>
                            <h2 className="fontA hidden font-regular mt-3 lg:mt-6 text-[#2a6442] text-3xl sm:text-4xl">
                              Intelligent Study Workflows
                            </h2>
                            <p className="poppins font-light mt-3 lg:w-[90%] text-lg sm:text-lg">
                              Transform raw financial data into structured, explainable risk insights.
Our platform actively evaluates behavior, documents, and macro factors
instead of relying on static credit scores.
                            </p>

                            <div className="poppins font-light flex gap-6 flex lg:flex-row mt-4 lg:mt-4 lg:w-[90%] ">
                              <div className="">
                                <h4 className="text-[#2a6442] font-light text-lg lg:text-xl ">
                                  Smart scoring
                                 <br /> & explanations{" "}
                                </h4>
                                <p className="mt-3 font-light text-white ">
                                 Risk probabilities, SHAP explanations, behavioral indicators, and
decision breakdowns generated and linked directly to financial inputs. </p>
                              </div>

                              <div className=" ">
                                <h4 className="text-[#2a6442] text-lg lg:text-xl">
                              Behavioral & macro <br />mapping{" "}
                                </h4>
                                <p className="mt-3   text-white ">
                                  Visual risk maps showing financial dependencies, behavioral signals,
macro exposure, and portfolio impact progression.
                                </p>
                              </div>
                            </div>

                          </div>




                          {/* <a href="./solutions-planitu">
                            <div
                              className="fontA mt-5 lg:mt-10 CTABUTTON glass-card cursor-pointer  z-2  flex  items-center justify-center !rounded-[16px] w-[160px] h-[53.6px]  "
                              style={{
                                backgroundColor: "rgb(3 , 255, 64 , 32%)",
                                border: "1px solid rgb(82 255 89)",
                              }}
                            >
                              <span>Explore More </span>
                            </div>
                          </a> */}

                        </div>

                        <div className="hidden px-3  w-[26%]    lg:block">
                          <img
                            className=" rounded-[30px]    overflow-hidden"
                            src="./img/elements/Revenue Maximisation.png"
                            alt=""
                          />
                        </div>
                      </div>

                      <div className="slide-huhh w-[95vw] xl:h-[448px]  xl:w-[1080px]   absolute slide-card flex flex-col lg:flex-row  lg:justify-between lg:items-center   mx-auto text-white px-5"
                        style={{ backgroundColor: "#02816B7D" }}
                      >
                        <div className=" lg:w-[60%] py-5 p-2 lg:p-5">
                          <div className="flex items-center gap-5   fontA">
                            <img src="./img/elements/deck-03.png" alt="" />
                            {/* slide-card-h2 */}
                            <h2 
                              className="  font-light text-2xl  "
                              style={{ lineHeight: "normal" }}
                            >
                              Adaptive Credit Intelligence & Human Collaboration
                            </h2>
                          </div>
                          <h2 className="hidden fontA font-regular mt-3 lg:mt-6 text-[#03FFB2] text-3xl sm:text-4xl">
                            Adaptive Learning <br className="hidden xl:block" />{" "}
                            & Collaboration
                          </h2>
                          <p className="font-light poppins mt-3 lg:w-[90%] text-lg sm:text-lg">
                           Risk analysis adapts to real-world behavior—not static credit scores.
Our platform combines predictive AI, behavioral signals, and human oversight in one unified decision system.
                          </p>

                          <div className="font-light poppins flex  gap-6 mt-4 lg:mt-4">
                            <div>
                              <h4 className="text-[#03FFB2] text-lg lg:text-xl">
                                Behavioral
 <br />Risk Analysis
                              </h4>
                              <p className="mt-3 text-white ">
                               AI-analyzed voice patterns, intent signals, and
psychological indicators extracted from
applicant conversations.
                              </p>
                            </div>

                            <div>
                              <h4 className="text-[#03FFB2] text-lg lg:text-xl">
                                Agent
  <br />Decision Console {" "}
                              </h4>
                              <p className="mt-3 text-white lg:w-[90%] ">
                               Unified dashboards combining financial data,
behavioral risk signals, and explainable
AI-backed decision insights.
                              </p>
                            </div>
                          </div>

                          {/* <a href="./solutions-rightsu">
                            <div
                              className="fontA CTABUTTON mt-2 glass-card cursor-pointer  z-2  flex  items-center justify-center !rounded-[16px] w-[160px] h-[53.6px]  "
                              style={{
                                backgroundColor: "rgb(3 , 255, 64 , 32%)",
                                border: "1px solid rgb(82 255 89)",
                              }}
                            >
                              <span>Explore More</span>
                            </div>
                          </a> */}
                        </div>

                        <div className="hidden px-4      w-[26%]  lg:block">
                          <img
                            className=" rounded-[30px]    overflow-hidden"
                            src="./img/elements/Content & Rights Management.jpg"
                            alt=""
                          />
                        </div>
                      </div>

                      <div className="slide-huhh w-[95vw] xl:h-[448px]   xl:w-[1080px]  md:h-fit  absolute slide-card flex flex-col lg:flex-row lg:justify-between lg:items-center  mx-auto text-white px-5  "
                        style={{ backgroundColor: "#134D7ABD" }}
                      >
                        <div className=" lg:w-[70%] py-5  p-2 lg:p-5">
                          <div className=" sm:h-fit lg:overflow-hidden">
                            <div className="flex items-center gap-5   fontA">
                              <img src="./img/elements/deck-03.png" alt="" />
                              {/* slide-card-h2 */}
                              <h2
                                className="   font-light text-2xl   "
                                style={{ lineHeight: "normal" }}
                              >
                                Multi-Layer Explainable AI Architecture
                              </h2>
                            </div>
                            <h2 className="fontA hidden font-regular mt-3 lg:mt-6 text-[#03FFB2] text-3xl sm:text-4xl">
                              Multi-Layer Retrieval<br /> Architecture
                            </h2>
                            <p className="font-light poppins mt-2 lg:w-[90%] text-lg sm:text-lg">
                            Accurate credit decisions require more than a traditional score.
Our platform uses a multi-layer AI system to ensure precision, transparency, and regulatory-grade explainability.
                               
                            </p>

                            <div className="poppins font-light flex gap-3 inner-scroll  overflow-scroll sm:overflow-hidden h-[220px] sm:h-fit mt-4  lg:mt-4">
                              <div className="lg:w-[500px]">
                                <h4 className="text-[#03FFB2]  text-lg lg:text-xl">
                                 4-layer Risk Modeling Engine
                                </h4>
                                



                                <p className=" mt-3 text-white ">
                                 Financial scoring model + Behavioral risk model+{" "}
                                  <br className="hidden lg:block" />
                                  Employer & sector analysis +{" "}
                                  <br className="hidden lg:block" />  Macro stress adjustment engine.
                                </p>
                              </div>

                              <div className="">
                                <h4 className="text-[#03FFB2]  text-lg lg:text-xl">
                                 Context-aware Risk Evaluation{" "}
                                </h4>
                                <p className=" mt-3 text-white ">
                                 Combines financial signals, behavioral intent, and
macro-economic context before final decision scoring.
                                </p>

                                {/* <h4 className="text-[#03FFB2]  text-lg lg:text-xl mt-2">
                                  Real-time workflow orchestration{" "}
                                </h4>
                                <p className=" mt-3 text-white ">
                                  Seamless integration across all broadcast
                                  touchpoints
                                </p> */}
                              </div>
                            </div>


                            {/* <div className=" hidden  overflow-scroll" >
                              <div className="" >
                                <h4 className="text-[#03FFB2]  text-lg lg:text-xl">
                                  Multi-Platform Content Orchestration
                                </h4>
                                <p className="  text-white ">
                                  Streamline linear and{" "}
                                  <br className="hidden lg:block" /> non-linear
                                  content with consistency
                                </p>
                              </div>

                              <div className="" >
                                <h4 className="text-[#03FFB2]  text-lg lg:text-xl">
                                  Predictive Performance Planning{" "}
                                </h4>
                                <p className="  text-white ">
                                  Fuse consumer research, planning rules,
                                  rights, and compliance into optimized
                                  schedules that predict audience performance.
                                </p>
                              </div>
                              <div className="" >
                                <h4 className="text-[#03FFB2]  text-lg lg:text-xl mt-2">
                                  Real-time workflow orchestration{" "}
                                </h4>
                                <p className="  text-white ">
                                  Seamless integration across all broadcast
                                  touchpoints
                                </p>
                              </div>
                            </div> */}
                          </div>

                          {/* <div className="lg:mt-[-10px]">
                            <a href="./solutions-broadview">
                              <div
                                className="fontA lg:mt-5 CTABUTTON glass-card cursor-pointer  z-2  flex   items-center justify-center !rounded-[16px] w-[160px] h-[53.6px]  "
                                style={{
                                  backgroundColor: "rgb(3 , 255, 64 , 32%)",
                                  border: "1px solid rgb(82 255 89)",
                                }}
                              >
                                <span>Explore More </span>
                              </div>
                            </a>
                          </div> */}
                        </div>

                        <div className="hidden    px-4 pr-3     w-[26%]    lg:block">
                          <img
                            className=" rounded-[30px]     overflow-hidden"
                            src="./img/elements/Broadcast Management.jpg"
                            alt=""
                          />
                        </div>
                      </div>

                      <div className="slide-huhh w-[95vw] xl:h-[448px]  xl:w-[1080px]   absolute slide-card flex flex-col lg:flex-row lg:justify-between lg:items-center  mx-auto text-white px-5  "
                        style={{ backgroundColor: "#0E4D47BD" }}
                      >
                        <div className=" py-5 lg:w-[50%]  p-2 lg:p-5">
                          <div className="flex items-center gap-5   fontA">
                            <img
                              className=""
                              src="./img/elements/deck-01.png"
                              alt=""
                            />
                            {/* slide-card-h2 */}
                            <h2
                              className="  font-light text-2xl  "
                              style={{ lineHeight: "normal" }}
                            >
                              Behavioral & Intent Intelligence
                            </h2>
                          </div>
                          <h2 className="hidden mt-14 sm:mt-0 fontA font-regular mt-3 lg:mt-6 text-[#03FFB2] text-3xl sm:text-4xl">
                            Learning Pattern Intelligence
                          </h2>
                          <p className="poppins  font-light mt-3 md:mt-3 lg:w-[500px] text-lg sm:text-lg">
                            Our platform doesn’t just score applicants—it understands behavioral intent and adapts risk evaluation accordingly.
                          </p>

                          <div className="poppins font-light  flex mt-4 lg:mt-4">
                            <div>
                              <h4 className="text-[#03FFB2] lg:w-[220px] text-lg lg:text-xl">
                                Intent behavior modeling{" "}
                              </h4>
                              <p className="mt-3 text-white lg:w-[260px]  ">
                                Tracks voice stress levels, response consistency,
financial literacy signals, and
behavioral risk indicators..
                              </p>
                            </div>

                            <div>
                              <h4 className="text-[#03FFB2] lg:w-[260px] text-lg lg:text-xl">
                              Adaptive risk adjustment layer
                              </h4>
                              <p className="mt-3 text-white ">
                               Automatically adjusts risk weighting,
flags inconsistencies, and
recalibrates scoring based on behavioral signals.
                              </p>
                            </div>
                          </div>
                          {/* <a className="" href="./contact">
                            <div
                              className="fontA  CTABUTTON mt-30  sm:mt-10    glass-card cursor-pointer  z-2  flex   items-center justify-center !rounded-[16px] w-[160px] h-[53.6px]  "
                              style={{
                                backgroundColor: "rgb(3 , 255, 64 , 32%)",
                                border: "1px solid rgb(82 255 89)",
                              }}
                            >
                              <span>Explore More</span>
                            </div>
                          </a> */}
                        </div>

                        <div className="hidden px-5     w-[26%]   lg:block">
                          <img
                            className=" rounded-[30px]    overflow-hidden"
                            src="./img/elements/AI Powered Dubbing.jpg"
                            alt=""
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-220"></div>

                  <div className="t-element hidden lg:hidden   md:h-[200px] md:scale-1">
                    <div className="bg-[#282828] ml-[-64px] h-[40px] w-[200px]"></div>
                  </div>
                </div>
              </section>

              <section className=" hidden  bg-white section-thatMatter text-black relative py-10 md:pb-25">
                <h1
                  data-speed="0.9"
                  className="fontA text-[#3b3b3b] font-medium text-center reveal-text reveal-text-3 my-10 md:my-15 text-5xl sm:text-6xl"
                >
                  <span>Numbers</span>
                  <span>That Matter</span>
                </h1>

                <div
                  data-speed="1.1"
                  className="flex flex-col lg:flex-row  lg:max-w-7xl   mx-auto px-6 gap-5   justify-between"
                >
                  {/* 250+ */}
                  <div className=" lg:w-[50%] flex flex-col gap-5  ">
                    <div className="  thatMatter-card-LU bg-[#272727]  rounded-[30px] flex  flex-col md:flex-row text-white poppins p-6  w-fit ">
                      <div>
                        <div className="flex  items-center">
                          <img src="/img/elements/element (6).png" alt="" />
                          <p className="ml-4 text-[#0FB381]">
                            Facts & <br /> Numbers
                          </p>
                        </div>

                        <h2
                          id=" "
                          className="  fontA font-regular mt-5 lg:mt-8  text-4xl  "
                        >
                          <Counter target={250} duration={1000} /> +
                        </h2>
                        <p className=" poppins font-light mt-1 lg:w-[80%]">
                          Channels powered across Digital Platforms and
                          Television
                        </p>
                      </div>
                      <div className="mt-5 flex items-center lg:mt-0">
                        <img src="/img/elements/element-03.png" alt="" />
                      </div>
                    </div>

                    <div className="  hidden md:flex  thatMatter-card-LD flex flex-row   justify-between gap-5">
                      <div className="bg-[#272727] rounded-[30px] flex text-white poppins p-6  w-fit ">
                        <div data-speed="0.94">
                          <div className="flex items-center">
                            <img src="/img/elements/Group 329.png" alt="" />
                            <p className="ml-4  text-[#02EAF3] ">
                              Facts & <br /> Numbers
                            </p>
                          </div>

                          <h2 className="fontA  font-regular mt-8 text-4xl   ">
                            <Counter target={25} duration={800} />+ Years
                          </h2>
                          <p className="poppins font-light mt-1">
                            Of solving media’s toughest challenges
                          </p>
                        </div>
                      </div>

                      <div className="bg-[#272727] rounded-[30px] flex text-white poppins p-6  w-fit ">
                        <div>
                          <div className="flex items-center">
                            <img src="/img/elements/element (8).png" alt="" />
                            <p className="ml-4 text-[#FEF202]  ">
                              Facts & <br /> Numbers
                            </p>
                          </div>

                          <h2 className="fontA  font-regular  mt-8  text-4xl ">
                            $<Counter target={5} duration={800} /> Billion
                          </h2>
                          <p className=" poppins font-light mt-1">
                            Revenue managed annually
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  {/* below 250+ */}
                  <div

                    className=" lg:w-[50%] flex flex-col gap-5"
                  >
                    <div data-speed="0.92" className="hidden md:flex  thatMatter-card-RU   justify-between gap-5">
                      <div className="bg-[#272727] rounded-[30px] flex text-white poppins p-6  w-fit ">
                        <div>
                          <div className="flex items-center">
                            <img src="/img/elements/element (9).png" alt="" />
                            <p className="ml-4 text-[#9F57FA] ">
                              Facts & <br /> Numbers
                            </p>
                          </div>

                          <h2 className="fontA   mt-8 text-4xl   ">
                            <Counter target={5} duration={800} step={1} />{" "}
                            Million
                          </h2>
                          <p className="mt-1 poppins font-light">
                            Rights Clearances managed annually
                          </p>
                        </div>
                      </div>

                      <div className="bg-[#272727] rounded-[30px] flex text-white poppins p-6  w-fit ">
                        <div>
                          <div className="flex items-center">
                            <img src="/img/elements/Group 332.png" alt="" />
                            <p className="ml-4 text-[#EE5988] ">
                              Facts & <br /> Numbers
                            </p>
                          </div>

                          <h2 className="fontA font-medium  mt-8 text-4xl  ">
                            <Counter target={125} step={10} duration={800} />
                            ,000
                          </h2>
                          <p className="mt-1 poppins font-light">
                            Hours of Content Planned
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className=" md:hidden flex  flex-wrap my-2 gap-4">
                      <div className="w-full flex justify-between">
                        <div className="flex-shrink-0  bg-[#125050] rounded-[30px] flex text-white poppins p-4  w-[48%] ">
                          <div>
                            <div className="flex items-center">
                              <img
                                className="w-[30px]"
                                src="/img/elements/element (9).png"
                                alt=""
                              />
                              <p className="ml-2 text-[#9F57FA]  text-sm ">
                                Facts & <br /> Numbers
                              </p>
                            </div>

                            <h2 className="fontA   my-3 text-3xl sm:text-4xl   ">
                              <Counter target={5} duration={600} step={1} />{" "}
                              Million
                            </h2>
                            <p className="mt-1 text-sm  poppins font-light">
                              Rights Clearances managed annually
                            </p>
                          </div>
                        </div>

                        <div className="flex-shrink-0  bg-[#125050] rounded-[30px] flex text-white poppins p-4  w-[48%] ">
                          <div>
                            <div className="flex items-center">
                              <img
                                className="w-[30px]"
                                src="/img/elements/Group 332.png"
                                alt=""
                              />
                              <p className="ml-2 text-sm text-[#EE5988] ">
                                Facts & <br /> Numbers
                              </p>
                            </div>

                            <h2 className="fontA font-medium  my-3 text-3xl sm:text-4xl  ">
                              <Counter target={125} step={10} duration={800} />
                              ,000
                            </h2>
                            <p className="mt-1  text-sm poppins font-light">
                              Hours of Content Planned
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="flex justify-between">
                        <div className="flex-shrink-0  bg-[#125050] rounded-[30px] flex text-white poppins p-4  w-[48%] ">
                          <div>
                            <div className="flex items-center">
                              <img
                                className="w-[30px]"
                                src="/img/elements/element (8).png"
                                alt=""
                              />
                              <p className="ml-2 text-sm text-[#FEF202] ">
                                Facts & <br /> Numbers
                              </p>
                            </div>

                            <h2 className="fontA font-regular  my-3 text-3xl sm:text-4xl ">
                              $<Counter target={5} duration={800} /> Billion
                            </h2>
                            <p className="mt-1 text-sm poppins font-light">
                              Revenue managed annually
                            </p>
                          </div>
                        </div>

                        <div className="flex-shrink-0  bg-[#125050] rounded-[30px] flex text-white poppins p-4  w-[48%] ">
                          <div className="">
                            <div className="flex items-center">
                              <img
                                className="w-[30px]"
                                src="/img/elements/Group 329.png"
                                alt=""
                              />
                              <p className="ml-2 text-sm  text-[#02EAF3] ">
                                Facts & <br /> Numbers
                              </p>
                            </div>

                            <h2 className="fontA font-regular  my-3 text-3xl sm:text-4xl  ">
                              <Counter target={25} duration={1000} step={1} />+{" "}
                              <br /> Years
                            </h2>
                            <p className="mt-1 text-sm poppins font-light">
                              {" "}
                              Of solving media’s toughest challenges
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div data-speed="0.92" className="md:mt-5 thatMatter-card-RD   bg-[#272727]   flex-col-reverse items-end md:items-start md:flex-row justify-between rounded-[30px] flex text-white poppins p-6    ">
                      <div className="w-[100%] md:w-fit">
                        <h2 className="fontA font-medium w-fit  text-4xl  ">
                          Ready to <br /> scale?
                        </h2>

                        <CTA className="w-[160px] mt-9" text="Let's Talk" />
                      </div>

                      <div data-speed="0.94">
                        <img
                          width="170px"
                          src="/img/elements/element (7).png"
                          alt=""
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </section>


              {/* <div className="anyClassWrap h-[100vh]" >
                Hello bRUHHH
                <div className="anyCLASS" >
                  This is different
                </div>
              </div> */}

              <Recognition />
                {/* this sections i have no understanding of why it is here  and what it is doing  but i am keeping it as it is cause it is working fine and i dont want to break anything by changing it */}
              <div className="block lg:hidden hidden " >
                <section ref={sectionRef} className="bg-[#E0DFDC] flex justify-center  h-[100vh]  overflow-hidden  text-white">
                  <div
                    ref={textRef}
                    className="relative my-20 mt-25 lg:mt-0">


                    <div className="   text-[60px] md:text-[120px] lg:text-[160px] 2xl:text-[200px]" >
                      <div className="fontA    flex overflow-hidden font-bold text-[#c2c2c2] ">
                        {text.split("").map((char, i) => (
                          <span key={i} className=" spanElement inline-block opacity-0">
                            {char}
                          </span>
                        ))}
                      </div>


                      <div className="revealTextBoxMob absolute top-0 fontA flex overflow-hidden   font-bold text-[#272727]" style={{ clipPath: "inset(0 0 0 100%)" }} >
                        {text.split("").map((char, i) => (
                          <span key={i} className="inline-block  ">
                            {char}
                          </span>
                        ))}

                      </div>

                    </div>


                    <div className=" revealTextCards mt-5  gap-40 flex "  >
                      <div className=" relative mx-auto items-center flex flex-col gap-5  " >

                        <div className="  bg-transparent absolute top-0 text-[#272727] flex-shrink-0   group scroll-fade-card    transtion-all duration-400   ">
                          <div className="impact-card-wrapper h-[400px] w-[300px] overflow-hidden text-white   glass-card  !rounded-[27px] p-1.5" style={{ backgroundColor: "#134C78" }} >

                            <div className="impact-card overflow-hidden p-6  flex flex-col justify-between items-start" style={{ height: "fit-content" }} >
                              <img className="child w-[100%] z-[-1] h-[60%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opaci ty-00" src="/img/elements/cardimg.png" alt="" />

                              <img className="child hidden transition-opacity duration-400 group-hover:opac ity-0 " src="/img/elements/element (11).svg" alt="" />
                              <div>
                                <h2 className="   z-1 mt-0 mb-2 fontA font-regular text-2xl transition-all duration-700 group-hover:tex t-[#F6F1E6]" >
                                  BroadcastPro Manufacturer Awards 2025 - Best in AI Winner</h2>
                                <p className="z-1 font-light text-lg poppins">
                                  Plan-itU recognized for transformative  AI-driven automation  in linear broadcast content planning and operations. </p>

                              </div>
                            </div>

                          </div>
                        </div>

                        <div className="  bg-transparent absolute top-0 text-[#272727] flex-shrink-0   group scroll-fade-card    transtion-all duration-400  "
                        >
                          <div className="impact-card-wrapper h-[400px] w-[300px] overflow-hidden text-white  glass-card  !rounded-[27px] p-1.5" style={{ backgroundColor: "#80B556" }} >

                            <div className="impact-card overflow-hidden p-6  flex flex-col justify-between items-start" style={{ height: "fit-content" }} >
                              <img className="child w-[100%] z-[-1] h-[70%] fadeout-mask absolute left-0 top-0 transition-opacity opacity-0 duration-700 group-hover:opacity-00" src="/img/elements/cardimg-sony.png" alt="" />

                              <img className="child transition-opacity hidden duration-400 group-hover:opac ity-0 " src="/img/elements/purple.svg" alt="" />
                              <div>
                                <h2 className=" z-1  mb-2 fontA font-regular text-2xl transition-all duration-700 group-hover:t ext-[#F6F1E6]" >
                                  Powering Sony Pictures Networks India Cloud Journey</h2>
                                <p className="z-1 font-light text-lg poppins">
                                  Successfully implemented BroadView in  AWS cloud for SPNI , setting new standards for enterprise-scale migrations.</p>

                              </div>
                            </div>

                          </div> 
                        </div>

                        <div className="  bg-transparent absolute top-0 text-[#272727] flex-shrink-0   group  scroll-fade-card  transtion-all duration-400  " >
                          <div className="impact-card-wrapper h-[400px] w-[300px]  overflow-hidden   glass-card  !rounded-[27px] p-1" style={{ backgroundColor: "#027B66" }} >

                            <div className="impact-card overflow-hidden p-6  text-white flex flex-col justify-between items-start " style={{ height: "fit-content" }}  >
                              <div>
                                <h2 className="   z-1 mt-8 mb-4 fontA font-regular text-2xl transition-all duration-700 group-hover:t ext-[#F6F1E6]" >
                                  Industry <br /> Leadership </h2>
                                <p className="z-1 font-light text-lg poppins">
                                  Featured at NAB 2024, The Broadcast Bridge, and  Broadcast & CableSat   for pioneering solutions that redefine media operations. </p>

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
              </div>

              <section className="bg-[#272727]  h-[100vh] flex items-center justify-center textScroll-cont">
                <div className="flex flex-col items-center ">
                  <div className="uppercase flex flex-col justify-center items-center text-[#CCCAC4] fontA  text-center font-medium  text-3xl sm:text-7xl">
                    <div className="mt-10 flex flex-col  items-between ">
                      <h1 className="textScroll-01  text-4xl lg:text-[7vw] xl:text-[10vh] 2xl:text-[13.5vh] gradient-text-textSlide">
                        Let’s Build your
                      </h1>

                      <div className="flex w-fit items-center gap-5  textScroll-01b gradient-text-textSlide">
                        <div className="w-[20px] lg:w-[4vw]  2xl:w-[9.5vh]">
                          <svg
                            width="100%"
                            height="100%"
                            viewBox="0 0 76 76"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path
                              d="M60.2715 49.6641V20.334H75.2715V75.2715H17.0459V60.2715H49.6641L0 10.6074L10.6074 0L60.2715 49.6641Z"
                              fill="#1BB23C"
                            />
                          </svg>

                        </div>

                        <span className="  text-4xl lg:text-[7vw] xl:text-[10vh] 2xl:text-[13.5vh]   ">FORESIGHT</span>
                      </div>
                    </div>

                    <div className=" mt-[4vh] ">
                      <h1 className="textScroll-02  md:text-[4vw] xl:text-[8vh] 2xl:text-[10vh]">
                        Let’s explore how <br className="md:hidden" /> YOU can
                      </h1>
                    </div>

                    <div className="flex items-center lg:gap-6 textScroll-04">

                      <div className="hidden md:block w-[20px] lg:w-[40px]">
                        <svg
                          width="100%"
                          height="100%"
                          viewBox="0 0 76 76"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M60.2715 49.6641V20.334H75.2715V75.2715H17.0459V60.2715H49.6641L0 10.6074L10.6074 0L60.2715 49.6641Z"
                            fill="#CCCAC4"
                          />
                        </svg>

                      </div>
                      <h1 className=" md:text-[4vw] xl:text-[8vh] 2xl:text-[10vh]">TURN CREDIT DATA</h1>
                    </div>

                    <div className="">
                      <h1 className="textScroll-05  md:text-[4vw] xl:text-[8vh] 2xl:text-[10vh]">
                        into{" "}
                        <span className="text-[#3AFF16]">EXPLAINABLE INTELLIGENCE,</span>{" "}
                        so
                      </h1>
                    </div>

                    <div className="">
                      <h1 className="textScroll-06 md:text-[4vw] xl:text-[8vh] 2xl:text-[10vh] ">
                        you  {" "}
                        <span className="text-[#3AFF16]"> stay ahead</span>
                      </h1>
                    </div>
                  </div>

                  <CTA className="w-[160px] mt-10" text="LOGIN" />
                </div>
              </section>
            </div>
          </div>
        </div>
              {/* <Footer bgColor="" /> */}
      </div>
    );
  }
}
