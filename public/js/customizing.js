
$(window).load(function(){
	$(".menu-icon").children("button").click(function(){
		$(".left-side").addClass("on");
	});
	$(".left-side").children(".close").click(function(){
		$(".left-side").removeClass("on");
	});


    $(".more_btn").click(function(){
        var answer = $(this).children("ul");
        if(answer.css("display")=="none"){
            answer.slideDown();  
        }else{
            answer.slideUp();
        }
       
    });

});


// $(window).scroll(function(){
// 	var scrTop = $(window).scrollTop();
// 	if(scrTop++){
// 		$(".custom-nav").fadeOut(300);
// 	}else if(scrTop--){
// 		$(".custom-nav").fadeIn(300);
// 	}
// });



$(document).ready(function() {

// Hide Header on on scroll down
var didScroll;
var lastScrollTop = 0;
var delta = 5;
var navbarHeight = $('.custom-nav').outerHeight();

$(window).scroll(function(event){
    didScroll = true;
});

setInterval(function() {
    if (didScroll) {
        hasScrolled();
        didScroll = false;
    }﻿
}, 0);

function hasScrolled() {
    var st = $(this).scrollTop();
    
    if(st <= 0){
    	$(".custom-nav").addClass("top");
    }else{
    	$(".custom-nav").removeClass("top");
    }

    // Make sure they scroll more than delta
    if(Math.abs(lastScrollTop - st) <= delta)
        return;
    
    // If they scrolled down and are past the navbar, add class .nav-up.
    // This is necessary so you never see what is "behind" the navbar.
    if (st > lastScrollTop && st > navbarHeight){
        // Scroll Down
        $(".custom-nav").fadeOut(300);
    } else {
        // Scroll Up
        if(st + $(window).height() < $(document).height()) {
            $(".custom-nav").fadeIn(300);
        }
    }
    
    lastScrollTop = st;
}

});
