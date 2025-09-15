document.addEventListener('DOMContentLoaded', () => {
    const animateOnScroll = () => {
        const cards = document.querySelectorAll('.review-card');
        cards.forEach(card => {
            const cardTop = card.getBoundingClientRect().top;
            if (cardTop < window.innerHeight * 0.8) {
                card.classList.add('visible');
            }
        });
    };

  const buttons = document.querySelectorAll('.filter-btn');
  const items   = document.querySelectorAll('.review-item');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const type = btn.dataset.type; 
      items.forEach(item => {
        if (type === 'all' || item.classList.contains(type)) {
          item.style.display = '';     
        } else {
          item.style.display = 'none'; 
        }
      });
    });
  });


    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); 

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

const themeToggle = document.querySelector('.theme-toggle');
const htmlElement = document.documentElement;

function toggleTheme() {
    htmlElement.classList.toggle('light-theme');
    const isLight = htmlElement.classList.contains('light-theme');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    themeToggle.innerHTML = isLight ? 
        '<i class="fas fa-sun"></i>' : 
        '<i class="fas fa-moon"></i>';
}

const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') {
    htmlElement.classList.add('light-theme');
    themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
}

themeToggle.addEventListener('click', toggleTheme);

document.addEventListener('DOMContentLoaded', () => {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.cyber-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        observer.observe(card);
    });
});


    const loadMoreContent = () => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    console.log('Загрузка новых отзывов...');
                }
            });
        });

        const lastCard = document.querySelector('.review-card:last-child');
        if (lastCard) observer.observe(lastCard);
    };

    loadMoreContent();

    document.querySelectorAll('.review-card').forEach(card => {
        card.addEventListener('click', () => {
            const reviewId = card.dataset.reviewId;
            if (reviewId) {
                window.location.href = `/reviews/${reviewId}`;
            }
        });
    });
});