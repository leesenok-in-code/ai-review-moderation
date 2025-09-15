document.addEventListener('DOMContentLoaded', function() {

    const themeSwitcher = document.getElementById('theme-switcher');
    const body = document.body;
    
    const savedTheme = localStorage.getItem('theme') || 'light';
    body.setAttribute('data-theme', savedTheme);
    themeSwitcher.checked = savedTheme === 'dark';

    themeSwitcher.addEventListener('change', function() {
        const theme = this.checked ? 'dark' : 'light';
        body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    });

    const form = document.getElementById('reviewForm');
    const jurSection = document.getElementById('jur_section');
    const personTypeRadios = document.querySelectorAll('input[name="person_type"]');
    const innInput = document.getElementById('inn');
    const fetchCompanyBtn = document.getElementById('fetchCompanyBtn');
    const submitBtn = document.getElementById('submitBtn');
    const notification = document.getElementById('notification');
    const fileInput = document.getElementById('document');
    const fileNameSpan = document.getElementById('file-name');

    
    function validateINN(inn) {
        return /^\d{10,12}$/.test(inn);
    }

    function toggleJurSection() {
        const isJur = document.querySelector('input[name="person_type"]:checked').value === 'jur';
        jurSection.classList.toggle('hidden', !isJur);
        innInput.toggleAttribute('required', isJur);
        
        if (isJur) {
            innInput.setAttribute('aria-required', 'true');
        } else {
            innInput.removeAttribute('aria-required');
        }
    }
const sellerIsCompanyCheckbox = document.getElementById('seller_is_company');
const sellerCompanySection = document.getElementById('seller_company_section');
const fetchSellerCompanyBtn = document.getElementById('fetchSellerCompanyBtn');
const sellerInnInput = document.getElementById('seller_inn');

function toggleSellerCompanySection() {
  sellerCompanySection.classList.toggle('hidden', !sellerIsCompanyCheckbox.checked);
}


async function fetchSellerCompanyData() {
  const inn = sellerInnInput.value.trim();
  
  if (!validateINN(inn)) {
    showNotification('error', 'Введите корректный ИНН');
    return;
  }

  try {
    fetchSellerCompanyBtn.disabled = true;
    const response = await fetch(`/api/company/${inn}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Ошибка сервера');
    }
    
    const data = await response.json();
    
    if (data.success) {
      document.getElementById('seller_company_name').value = data.data.name || '';
      document.getElementById('seller_ogrn').value = data.data.ogrn || '';
      document.getElementById('seller_company_address').value = data.data.address || '';
      document.getElementById('seller_company_city').value = data.data.city || '';
    } else {
      showNotification('error', 'Компания не найдена');
    }
  } catch (error) {
    showNotification('error', error.message);
    console.error('Ошибка:', error);
  } finally {
    fetchSellerCompanyBtn.disabled = false;
  }
}

const scenarioSelect = document.getElementById('scenario');
const scenarioDetails = {
  prepayment: document.getElementById('prepayment-details'),
  noprepayment: document.getElementById('noprepayment-details'),
  other: document.getElementById('other-details')
};

function updateScenarioDetails() {
  Object.values(scenarioDetails).forEach(detail => {
    detail.classList.add('hidden');
  });
  
  const selectedValue = scenarioSelect.value;
  if (selectedValue && scenarioDetails[selectedValue]) {
    scenarioDetails[selectedValue].classList.remove('hidden');
  }
}

updateScenarioDetails();

scenarioSelect.addEventListener('change', updateScenarioDetails);

sellerIsCompanyCheckbox.addEventListener('change', toggleSellerCompanySection);
fetchSellerCompanyBtn.addEventListener('click', fetchSellerCompanyData);

async function handleSubmit(e) {
  e.preventDefault();
  
  if (sellerIsCompanyCheckbox.checked) {
    const sellerCompanyName = document.getElementById('seller_company_name').value;
    if (!sellerCompanyName.trim()) {
      showNotification('error', 'Пожалуйста, заполните данные компании');
      return;
    }
  }
}

const imagesInput = document.getElementById('work_images');
const imagesNamesSpan = document.getElementById('images-names');
const imagesPreviewDiv = document.getElementById('images-preview');

imagesInput.addEventListener('change', function() {
  imagesPreviewDiv.innerHTML = '';
  
  if (this.files.length > 0) {
    imagesNamesSpan.textContent = this.files.length + ' файл(ов) выбрано';
    
    for (let i = 0; i < this.files.length; i++) {
      const file = this.files[i];
      const reader = new FileReader();
      
      reader.onload = function(e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.style.maxWidth = '100px';
        img.style.maxHeight = '100px';
        img.style.margin = '5px';
        imagesPreviewDiv.appendChild(img);
      }
      
      reader.readAsDataURL(file);
    }
  } else {
    imagesNamesSpan.textContent = 'Файлы не выбраны';
  }
});

imagesInput.addEventListener('change', function() {
  if (this.files.length > 5) {
    alert('Можно загрузить не более 5 изображений');
    this.value = '';
    imagesNamesSpan.textContent = 'Файлы не выбраны';
    imagesPreviewDiv.innerHTML = '';
  }
});

    function showNotification(type, message) {
        notification.className = `notification ${type} show`;
        notification.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}" aria-hidden="true"></i>
            <span>${message}</span>
        `;
        setTimeout(() => {
            notification.classList.remove('show');
        }, 5000);
    }

   async function fetchCompanyData() {
    const inn = innInput.value.trim();
    
    if (!validateINN(inn)) {
        showNotification('error', 'Введите корректный ИНН');
        return;
    }

    try {
        fetchCompanyBtn.disabled = true;
        const response = await fetch(`/api/company/${inn}`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Ошибка сервера');
        }
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('company_name').value = data.data.name || '';
            document.getElementById('ogrn').value = data.data.ogrn || '';
            document.getElementById('address').value = data.data.address || '';
            document.getElementById('city').value = data.data.city || '';
        } else {
            showNotification('error', 'Компания не найдена');
        }
    } catch (error) {
        showNotification('error', error.message);
        console.error('Ошибка:', error);
    } finally {
        fetchCompanyBtn.disabled = false;
    }
}

    async function handleSubmit(e) {
        e.preventDefault();
        
        if (!document.getElementById('agree').checked) {
            showNotification('error', 'Необходимо подтвердить согласие');
            return;
        }

        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        try {
            const formData = new FormData(form);
            
            const response = await fetch('/', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Ошибка сервера');
            
            if (result.success) {
                showNotification('success', 'Отзыв успешно отправлен на модерацию!');
                form.reset();
                fileNameSpan.textContent = 'Файл не выбран';
                setTimeout(() => window.location.href = '/reviews', 2000);
            }
        } catch (error) {
            console.error('Ошибка отправки:', error);
            showNotification('error', error.message || 'Ошибка при отправке формы');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    }

    personTypeRadios.forEach(radio => {
        radio.addEventListener('change', toggleJurSection);
        radio.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') radio.click();
        });
    });

    fetchCompanyBtn.addEventListener('click', fetchCompanyData);
    form.addEventListener('submit', handleSubmit);
    fileInput.addEventListener('change', () => {
        fileNameSpan.textContent = fileInput.files[0]?.name || 'Файл не выбран';
    });
    
    toggleJurSection();
});