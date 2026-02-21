"use client";
import React, { useState, useEffect, useRef } from "react";
import GlassEffect from "./DEM";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { usePopup } from '@/context/PopupContext';

type NavbarProps = {
    menu: boolean;
    setMenu: any;
};

export default function Navbar() {
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [dropdownOpenedBy, setDropdownOpenedBy] = useState<"hover" | "click" | null>(null);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);
    const [menuHidden, setMenuHidden] = useState(false);
    const { openPopup } = usePopup();

    const [mobNav, setMobNav] = useState<false | true | "solutions">(false);
    const [mobileNav, setMobileNav] = useState(false)
    const mobNavRef = useRef<HTMLDivElement>(null); // ✅ ref for mobile menu

    const [isHoveringDropdown, setIsHoveringDropdown] = useState(false);
    const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        gsap.registerPlugin(ScrollTrigger);

        const handleScroll = () => {
            const scrollY = window.scrollY;
            // const vh100 = window.innerHeight;
            const vh100 = 40

            if (scrollY > vh100) {
                if (!menuHidden) {
                    setMenuHidden(true);
                }
            }
        };

        window.addEventListener("scroll", handleScroll);

        const handleClickOutside = (event: MouseEvent) => {
            // ✅ close desktop dropdown
            if (
                dropdownRef.current &&
                !dropdownRef.current.contains(event.target as Node) &&
                buttonRef.current &&
                !buttonRef.current.contains(event.target as Node)
            ) {
                setDropdownOpen(false);
                setDropdownOpenedBy(null);
            }

            // close mobile nav
            if (
                mobNav &&
                mobNavRef.current &&
                !mobNavRef.current.contains(event.target as Node)
            ) {
                setMobNav(false);
            }
        };

        document.addEventListener("mousedown", handleClickOutside);

        return () => {
            window.removeEventListener("scroll", handleScroll);
            document.removeEventListener("mousedown", handleClickOutside);
            if (hoverTimeoutRef.current) {
                clearTimeout(hoverTimeoutRef.current);
            }
        };
    }, [menuHidden, mobNav]);

    useEffect(() => {
        if (!dropdownOpen) {
            setDropdownOpenedBy(null);
            setIsHoveringDropdown(false);
        }
    }, [dropdownOpen]);

    return (
        <div className="navbar fixed z-[12] top-5 w-[100vw]" data-speed="0">
            {/* mobile navbar */}
            <div className="fontA lg:hidden w-[94vw] mx-auto" >
                <div className="w-[100%] flex justify-between" >
                    <div className=" w-[140px] h-[62px] navMobLeft  ">
                        <a className="CTABUTTON" href="./">
                            <GlassEffect>
                                <img src="/img/logo-mobile.png" alt="" width={140} />
                            </GlassEffect>
                        </a>
                    </div>
                    <div className="flex gap-2 navMobRight" >
                        <div onClick={() => openPopup("meeting")} className="CTABUTTON cursor-pointer   w-[140px] h-[62px] flex items-center " >
                            <GlassEffect height="62px" >
                                <span className="text-[16px]   text-[#959595] ">Book A Demo</span>
                            </GlassEffect>
                        </div>
                        <div className=" w-[60px] h-[62px] "  onClick={() => { setMobNav( !mobileNav    ) ; setMobileNav(prev => !prev)  } } >
                            <GlassEffect> 
                                <img width={20} className="py-4" src="/img/menu-dot.svg" alt="" />
                            </GlassEffect>
                        </div>
                    </div>
                </div>

                {mobNav && (
                    <div
                        ref={mobNavRef} // ✅ ref applied here
                        className="absolute top-[70px] left-0 w-full z-40 flex justify-center"
                    >
                        <div className="relative w-[92%] max-w-sm overflow-hidden rounded-2xl bg-[#f0f8ff00] backdrop-blur-md shadow-xl border border-[#7f7f7f6b]" style={{background:"linear-gradient(135deg, rgb(162 89 255 / 24%) 0%, rgb(107 0 102) 100%)"}}>
                            {/* Screens container */}
                            <div
                                className={`flex w-[200%] transition-transform duration-500 ease-in-out ${mobNav === "solutions" ? "-translate-x-1/2" : "translate-x-0"
                                    }`}
                            >
                                {/* --- Screen 1 --- */}
                                <div className="w-1/2 h-fit p-3">
                                    <div className="  flex flex-col gap-4 p-3 rounded-[10px]" >
                                        <GlassEffect>
                                            <button
                                                onClick={() => setMobNav("solutions")}
                                                className="flex items-center justify-between text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition"
                                            >
                                                <span>Study</span>
                                                <img
                                                    src="/img/rightarrow.svg"
                                                    alt="→"
                                                    className="w-4 ml-3 h-4 opacity-70"
                                                />
                                            </button>
                                        </GlassEffect>

                                        <GlassEffect>
                                            <a href="/about" className="flex" >
                                                <button className="text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition">
                                                    Company
                                                </button>
                                            </a>
                                        </GlassEffect>

                                        <GlassEffect>
                                            <a href="/contact" className="flex" >
                                                <button className="text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition">
                                                    Contact
                                                </button>
                                            </a>
                                        </GlassEffect>
                                    </div>
                                </div>

                                {/* --- Screen 2 --- */}
                                <div className="w-1/2 h-fit p-3">
                                    <div className="  flex flex-col gap-4 p-3 rounded-[10px]" >
                                        <GlassEffect>
                                            <button
                                                onClick={() => setMobNav(true)}
                                                className="flex items-center gap-2 text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition"
                                            >
                                                <span>← Back</span>
                                            </button>
                                        </GlassEffect>

                                        <GlassEffect>
                                            <a href="/solutions-broadview" className="flex" >
                                                <button className="text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition">
                                                    BroadView
                                                </button>
                                            </a>
                                        </GlassEffect>

                                        <GlassEffect>
                                            <a href="/solutions-rightsu" className="flex">
                                                <button className="text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition">
                                                    RightsU
                                                </button>
                                            </a>
                                        </GlassEffect>

                                        <GlassEffect>
                                            <a href="/solutions-planitu" className="flex">
                                                <button className="text-left text-lg text-[#d9d9d9] w-full hover:opacity-80 transition">
                                                    Plan-itU
                                                </button>
                                            </a>
                                        </GlassEffect>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

                  {/* desktop navbar */}
            <div className="poppins relative hidden lg:flex justify-center w-screen mx-auto">
                <div className="absolute top-0 h-[54px] navMidComp lg:block mx-auto lg:w-5xl xl:w-7xl lg:px-5   text-white fontA text-light flex">
                    <div className="flex h-[100%] gap-3 w-[100%] justify-between">
                        <div className="w-[200px]   ">
                            <a className="h-[54px] CTABUTTON" href="./">
                                <GlassEffect padding="0rem 0.5rem" >
                                    <img  className="h-[46px]" src="/img/CredCat-Photoroom.png" alt="" />
                                </GlassEffect>  
                            </a>
                        </div>

                        <div className="flex gap-2 poppins text-[18px]">
                            {/* menu with dropdown */}
                            {!menuHidden && (
                                <div
                                    className="relative "
                                    ref={dropdownRef}
                                >
                                    <GlassEffect width={"400px"}>
                                        <div className="fontA CTABUTTON text-lg poppins text-white px-2 flex w-full justify-end gap-5 opacity-70 menu">
                                            <button
                                                ref={buttonRef}
                                                onClick={() => {
                                                    setDropdownOpen(!dropdownOpen);
                                                    setDropdownOpenedBy("click");
                                                }}
                                                onMouseEnter={() => {
                                                    if (hoverTimeoutRef.current) {
                                                        clearTimeout(hoverTimeoutRef.current);
                                                    }
                                                    setDropdownOpen(true);
                                                    setDropdownOpenedBy("hover");
                                                }}
                                                onMouseLeave={() => {
                                                    // Only close if not hovering over dropdown and opened by hover
                                                    if (dropdownOpenedBy === "hover") {
                                                        hoverTimeoutRef.current = setTimeout(() => {
                                                            if (!isHoveringDropdown) {
                                                                setDropdownOpen(false);
                                                            }
                                                        }, 150);
                                                    }
                                                }}
                                                className="cursor-pointer"
                                            >
                                                <span className="!text-[#959595]">Apply</span>
                                            </button>
                                            <a className="fontA CTABUTTON" href="/about">
                                                <span className="!text-[#959595] ">AgentConsole</span>
                                            </a>
                                            <a className="fontA CTABUTTON  " href="/contact">
                                                <span className="!text-[#959595] "> Simulate </span>   
                                            </a>
                                        </div>
                                    </GlassEffect>

                                    {/* functional dropdown */}
                                    {/* {dropdownOpen && (
                                        <div
                                            className="absolute top-full right-0 mt-2 z-50"
                                            onMouseEnter={() => {
                                                if (hoverTimeoutRef.current) {
                                                    clearTimeout(hoverTimeoutRef.current);
                                                }
                                                setIsHoveringDropdown(true);
                                            }}
                                            onMouseLeave={() => {
                                                setIsHoveringDropdown(false);
                                                // Close dropdown on mouse leave if opened by hover
                                                if (dropdownOpenedBy === "hover") {
                                                    hoverTimeoutRef.current = setTimeout(() => {
                                                        setDropdownOpen(false);
                                                    }, 150);
                                                }
                                            }}
                                        >
                                            <GlassEffect width={"320px"}>
                                                <ul className="flex rounded-[20px]  p-2 !text-white flex-col gap-2 text-white text-sm" style={{background:"linear-gradient(135deg, rgb(162 89 255 / 24%) 0%, rgb(107 0 102) 100%)"}}>
                                                    <li>
                                                        <a
                                                            href="/solutions-broadview "
                                                            className="hover:opacity-100 CTABUTTON hover:bg-[#e7e7e71f] p-2 rounded-[12px]  opacity-70  items-center flex gap-2 transition"
                                                        >
                                                            <img
                                                                className="hidden rounded-[10px]"
                                                                src="/img/dropdown.png"
                                                                alt=""
                                                            />
                                                            <div>
                                                                <h2 className="text-xl">BroadView</h2>
                                                                <p className="poppins text-[14px]">
                                                                    Comprehensive Broadcast Management for Modern
                                                                    Media Operations
                                                                </p>
                                                            </div>
                                                        </a>
                                                    </li>

                                                    <li>
                                                        <a
                                                            href="/solutions-rightsu"
                                                            className="hover:opacity-100 opacity-70 CTABUTTON  hover:bg-[#e7e7e71f]  rounded-[12px] p-2 items-center flex gap-2 transition"
                                                        >
                                                            <img
                                                                className="hidden rounded-[10px]"
                                                                src="/img/dropdown-2.png"
                                                                alt=""
                                                            />
                                                            <div>
                                                                <h2 className="text-xl">RightsU</h2>
                                                                <p className="poppins text-[14px]">
                                                                    Next-Generation Rights Management
                                                                </p>
                                                            </div>
                                                        </a>
                                                    </li>

                                                    <li>
                                                        <a
                                                            href="/solutions-planitu"
                                                            className="hover:opacity-100 opacity-70 CTABUTTON  hover:bg-[#e7e7e71f]  rounded-[12px] p-2 items-center flex gap-2 transition"
                                                        >
                                                            <img
                                                                className="hidden rounded-[10px]"
                                                                src="/img/dropdown-3.png"
                                                                alt=""
                                                            />
                                                            <div>
                                                                <h2 className="text-xl">Plan-itU</h2>
                                                                <p className="poppins text-[14px]">
                                                                    AI-Powered Content Planning & Operations
                                                                </p>
                                                            </div>
                                                        </a>
                                                    </li>
                                                </ul>
                                            </GlassEffect>
                                        </div>
                                    )} */}
                                </div>
                            )}

                            {menuHidden && (
                                <div
                                    className="w-[160px] cursor-pointer"
                                    onClick={() => {
                                        setMenuHidden((prev) => !prev);
                                    }}
                                >
                                    <GlassEffect width={"160px"}>
                                        <div className="CTABUTTON text-lg px-1 !text-[#959595] flex w-fit justify-end gap-5 opacity-70 menu">
                                            <span>Menu</span>
                                        </div>
                                    </GlassEffect>
                                </div>
                            )}

                            <div className="cursor-pointer" onClick={() => openPopup("meeting")} >
                                <GlassEffect width={"160px"}>
                                    <div className="text-lg fontA CTABUTTON !text-[#959595]   px-1 flex w-fit justify-end gap-5 opacity-70 menu">
                                        <span>Login</span>
                                    </div>
                                </GlassEffect>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
