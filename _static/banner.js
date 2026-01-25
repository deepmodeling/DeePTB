/* The initial version is taken from
https://github.com/pytorch/pytorch_sphinx_theme
under MIT license
rewrite to remove jQuery by ChatGPT
*/
window.mobileMenu = {
  bind: function () {
    document
      .querySelectorAll("[data-behavior='open-mobile-menu']")
      .forEach(function (element) {
        element.addEventListener("click", function (e) {
          e.preventDefault();
          document.querySelector(".mobile-main-menu").classList.add("open");
          document.body.classList.add("no-scroll");

          mobileMenu.listenForResize();
        });
      });

    document
      .querySelectorAll("[data-behavior='close-mobile-menu']")
      .forEach(function (element) {
        element.addEventListener("click", function (e) {
          e.preventDefault();
          mobileMenu.close();
        });
      });
  },

  listenForResize: function () {
    function resizeHandler() {
      if (window.innerWidth > 768) {
        mobileMenu.close();
      }
    }

    window.addEventListener("resize", resizeHandler);

    // Store the handler so it can be removed later
    this.resizeHandler = resizeHandler;
  },

  close: function () {
    document.querySelector(".mobile-main-menu").classList.remove("open");
    document.body.classList.remove("no-scroll");

    // Remove the resize event listener
    if (this.resizeHandler) {
      window.removeEventListener("resize", this.resizeHandler);
      this.resizeHandler = null;
    }
  },
};

document.addEventListener("DOMContentLoaded", function () {
  mobileMenu.bind();
});
